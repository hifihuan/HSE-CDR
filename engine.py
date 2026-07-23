"""
HSE-CDR training engine for HICO-DET and V-COCO evaluation.
"""

import glob
import json
import os
import time
import copy
import torch
import numpy as np
import scipy.io as sio
try:
    import wandb
except Exception:
    class _WandbStub:
        def init(self, *args, **kwargs):
            return None
        def log(self, *args, **kwargs):
            return None
        @property
        def run(self):
            return None
    wandb = _WandbStub()
from tqdm import tqdm
from collections import defaultdict

from utils.hico_text_label import hico_unseen_index
import utils.ddp as ddp
from utils.visualization import HOIVisualizer
from utils.eval_utils import image_matches_eval_subset, summarize_hico_ap
import cv2
try:
    from pocket.core import DistributedLearningEngine
    from pocket.utils import DetectionAPMeter, BoxPairAssociation
except Exception:
    import sys as _sys, os as _os
    for _k in list(_sys.modules.keys()):
        if _k == 'pocket' or _k.startswith('pocket.'):
            del _sys.modules[_k]


    _ROOT_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _CANDIDATE_POCKET_DIRS = [
        _os.path.join(_ROOT_DIR, "pocket"),
        _os.path.join(_ROOT_DIR, "..", "pocket"),
    ]
    for _p in _CANDIDATE_POCKET_DIRS:
        if _os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)

    from pocket.core import DistributedLearningEngine
    from pocket.utils import DetectionAPMeter, BoxPairAssociation
import datetime


class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')

        if seg[-1] == 'agent':
            return 0.

        else:
            return [0., 0., .1, .1, 0.]

from torch.cuda import amp
from pocket.ops import relocate_to_cuda, relocate_to_cpu

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117,test_loader=None,args=None, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.net = net
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.train_loader = dataloader
        self.test_loader = test_loader
        self.best_unseen = -1
        self.best_seen = -1
        self.args = args
        self._restore_best_metric_tracker()

        self._epoch_start_time = None
        if self.args.amp:
            self.scaler = amp.GradScaler(enabled=True)


        self.model_stats = self._compute_model_stats()


    def _on_end_iteration(self):

        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.inputs = relocate_to_cuda(self._state.inputs,ignore=True, non_blocking=True)
        self._state.targets = relocate_to_cuda(self._state.targets,ignore=True, non_blocking=True)

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        t_iter_mean = self._state.t_iteration.mean()
        t_data_mean = self._state.t_data.mean()

        it_sec = t_iter_mean + t_data_mean


        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            current_iter = self._state.iteration - num_iter * (self._state.epoch - 1)
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter./Remain.]: [{:.2f}s/{:.2f}s/{}]".format(
                self._state.epoch, self.epochs,
                str(current_iter).zfill(n_d),
                num_iter, running_loss, t_data, t_iter,  datetime.timedelta(seconds=(num_iter-current_iter)*it_sec)
            ))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_each_iteration(self):

        if self._epoch_start_time is None:
            self._epoch_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        self._state.net.train()
        with amp.autocast(enabled=self.args.amp):
            loss_dict = self._state.net(
                *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")


        accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        self._state.loss = sum(loss for loss in loss_dict.values()) / accumulation_steps

        if self.args.amp:
            self.scaler.scale(self._state.loss).backward()


            if (self._state.iteration - 1) % accumulation_steps == 0:
                if self.max_norm > 0:
                    self.scaler.unscale_(self._state.optimizer)
                    torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
                self.scaler.step(self._state.optimizer)
                self.scaler.update()
                self._state.optimizer.zero_grad(set_to_none=True)
        else:
            self._state.loss.backward()


            if (self._state.iteration - 1) % accumulation_steps == 0:
                if self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
                self._state.optimizer.step()
                self._state.optimizer.zero_grad(set_to_none=True)

    def _on_end_epoch(self):


        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()
        self.net.object_class_to_target_class = self.test_loader.dataset.dataset.object_class_to_target_class


        if self.args.dataset == 'vcoco':
            ret = self.cache_vcoco(self.test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            det_file = 'vcoco_cache/cache.pkl'
            b= vcocoeval._do_eval(ret, ovr_thresh=0.5)
            mAPs = {
                'sc2': b[1]
            }

            wandb.log(mAPs)
            return


        peak_train_mem_gb = None
        peak_train_mem_alloc_gb = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            dev = torch.cuda.current_device()
            peak_train_mem_gb = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
            peak_train_mem_alloc_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)


        ap_default = self.test_hico(self.test_loader, self.args, eval_mode='default')
        eval_stats = dict(getattr(self, '_last_eval_stats', None) or {})
        ap_known_object = self.test_hico(self.test_loader, self.args, eval_mode='known_object')


        ap = ap_default

        self.net.object_class_to_target_class = self.train_loader.dataset.dataset.object_class_to_target_class
        self.net.tp = None


        num_anno = torch.as_tensor(self.train_loader.dataset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        if self._rank == 0:
            mAPs = {'mAP': ap.mean() * 100,
                    'rare': ap[rare].mean() * 100,
                    'non-rare': ap[non_rare].mean() * 100,
                    'mAP_known_object': ap_known_object.mean() * 100,
                    'rare_known_object': ap_known_object[rare].mean() * 100,
                    'non-rare_known_object': ap_known_object[non_rare].mean() * 100
                    }
            mAPs['full'] = mAPs['mAP']

            print(
                f"The mAP is {ap.mean() * 100:.2f},"
                f" rare: {ap[rare].mean() * 100:.2f},"
                f" none-rare: {ap[non_rare].mean() * 100:.2f},"
            )
            print(
                f"Known Object mAP: {ap_known_object.mean() * 100:.2f},"
                f" rare: {ap_known_object[rare].mean() * 100:.2f},"
                f" none-rare: {ap_known_object[non_rare].mean() * 100:.2f},"
            )

            if self.args.zs:
                zs_hoi_idx = hico_unseen_index[self.args.zs_type]
                print(f'>>> zero-shot setting({self.args.zs_type}!!)')
                ap_unseen = []
                ap_seen = []
                for i, value in enumerate(ap):
                    if i in zs_hoi_idx:
                        ap_unseen.append(value)
                    else:
                        ap_seen.append(value)

                ap_unseen = torch.as_tensor(ap_unseen).mean()
                ap_seen = torch.as_tensor(ap_seen).mean()

                mAPs.update({"unseen": ap_unseen * 100, "seen": ap_seen * 100})
                print(
                    f"full mAP: {ap.mean() * 100:.2f}",
                    f"unseen: {ap_unseen * 100:.2f}",
                    f"seen: {ap_seen * 100:.2f}",
                )

            mAPs['epoch'] = self._state.epoch

            if self._epoch_start_time is not None:
                elapsed_sec = time.time() - self._epoch_start_time
                mAPs['training_time_min'] = elapsed_sec / 60.0

                self._epoch_start_time = None
            if peak_train_mem_gb is not None:
                mAPs['peak_gpu_mem_gb'] = peak_train_mem_gb
            if peak_train_mem_alloc_gb is not None:
                mAPs['peak_gpu_mem_alloc_gb'] = peak_train_mem_alloc_gb
            if 'infer_ms_per_image' in eval_stats:
                mAPs['infer_ms_per_image'] = eval_stats['infer_ms_per_image']
            if self._rank == 0 and getattr(self.args, 'log_epoch_costs', False):
                t_min = mAPs.get('training_time_min')
                infer_ms = mAPs.get('infer_ms_per_image')
                t_str = f"{t_min:.2f}" if t_min is not None else "n/a"
                g_str = f"{peak_train_mem_gb:.2f}" if peak_train_mem_gb is not None else "n/a"
                ga_str = f"{peak_train_mem_alloc_gb:.2f}" if peak_train_mem_alloc_gb is not None else "n/a"
                i_str = f"{infer_ms:.2f}" if infer_ms is not None else "n/a"
                print(f"[EpochCost] epoch={self._state.epoch} train={t_str} min, "
                      f"GPU={g_str} GB (reserved, ~nvidia-smi), active={ga_str} GB, infer={i_str} ms/img")

            if isinstance(getattr(self, "model_stats", None), dict):

                if "trainable_params_m" in self.model_stats and self.model_stats["trainable_params_m"] is not None:
                    mAPs["trainable_params_m"] = self.model_stats["trainable_params_m"]
                elif "trainable_params" in self.model_stats:
                    mAPs["trainable_params_m"] = float(self.model_stats["trainable_params"]) / 1e6

                if "total_params_m" in self.model_stats and self.model_stats["total_params_m"] is not None:
                    mAPs["total_params_m"] = self.model_stats["total_params_m"]
                elif "total_params" in self.model_stats:
                    mAPs["total_params_m"] = float(self.model_stats["total_params"]) / 1e6

                if "flops_g" in self.model_stats and self.model_stats["flops_g"] is not None:
                    mAPs["flops_g"] = self.model_stats["flops_g"]
                elif "flops" in self.model_stats and self.model_stats["flops"] is not None:
                    mAPs["flops_g"] = float(self.model_stats["flops"]) / 1e9
            self._write_epoch_metrics(mAPs)
            self.save_checkpoint(mAPs)
            wandb.log(mAPs)

    def _restore_best_metric_tracker(self):
        """Restore best-unseen (or best full mAP) from existing ckpt on disk."""
        if self._rank != 0 or not getattr(self, '_cache_dir', None):
            return
        for name in ('ckpt_best_unseen.pt', 'ckpt_best.pt'):
            path = os.path.join(self._cache_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                ckpt = torch.load(path, map_location='cpu')
                metrics = ckpt.get('metrics', {})
                score = metrics.get('unseen', metrics.get('full', metrics.get('mAP')))
                if score is not None:
                    self.best_unseen = float(score)
                    break
            except Exception:
                pass

    def save_checkpoint(self, metrics=None) -> None:
        """Save only the latest epoch and the best-unseen checkpoint."""
        if self._rank != 0:
            return

        os.makedirs(self._cache_dir, exist_ok=True)
        checkpoint = {
            'iteration': self._state.iteration,
            'epoch': self._state.epoch,
            'model_state_dict': self._state.net.module.state_dict(),
            'optim_state_dict': self._state.optimizer.state_dict(),
            'scaler_state_dict': self._state.scaler.state_dict(),
        }
        if self._state.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics

        last_path = os.path.join(self._cache_dir, 'ckpt_last.pt')
        torch.save(checkpoint, last_path)
        print(f"[Checkpoint] Saved latest epoch {self._state.epoch} -> {last_path}")

        if metrics is not None:
            use_unseen = bool(self.args and self.args.zs and 'unseen' in metrics)
            score_key = 'unseen' if use_unseen else 'full'
            score = metrics.get(score_key, metrics.get('mAP'))
            if score is not None and score > self.best_unseen:
                self.best_unseen = float(score)
                best_name = 'ckpt_best_unseen.pt' if use_unseen else 'ckpt_best.pt'
                best_path = os.path.join(self._cache_dir, best_name)
                torch.save(checkpoint, best_path)
                print(f"[Checkpoint] New best {score_key} {score:.2f} -> {best_path}")

        self._cleanup_per_epoch_ckpts()

    def _cleanup_per_epoch_ckpts(self):
        keep = {'ckpt_last.pt', 'ckpt_best_unseen.pt', 'ckpt_best.pt'}
        for path in glob.glob(os.path.join(self._cache_dir, 'ckpt_*.pt')):
            if os.path.basename(path) in keep:
                continue
            try:
                os.remove(path)
            except OSError:
                pass

    def _compute_model_stats(self):
        """
        统计 HSE-CDR 在当前 UV 设置下的参数量与 FLOPs。
        - trainable_params：参与训练的参数总量（Tr. Params）
        - total_params：模型全部参数总量（Tot. Params）
        - flops：单次前向推理的 FLOPs（如无法计算则为 None）
        """
        stats = {
            "trainable_params": None,
            "total_params": None,
            "flops": None,
        }

        def _to_m(num):
            if num is None:
                return None
            return float(num) / 1e6

        def _to_g(num):
            if num is None:
                return None
            return float(num) / 1e9

        try:
            total_params = 0
            trainable_params = 0
            for p in self.net.parameters():
                n = p.numel()
                total_params += n
                if p.requires_grad:
                    trainable_params += n
            stats["total_params"] = total_params
            stats["trainable_params"] = trainable_params
        except Exception as e:
            if self._rank == 0:
                print(f"[ModelStats] Failed to count parameters: {e}")


        try:
            if self._rank == 0:
                from thop import profile

                def _move_to_device(obj, device):

                    if torch.is_tensor(obj):
                        return obj.to(device, non_blocking=True)
                    if isinstance(obj, dict):
                        return {k: _move_to_device(v, device) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        moved = [_move_to_device(v, device) for v in obj]
                        return type(obj)(moved)
                    return obj


                example_batch = next(iter(self.train_loader))


                example_images = relocate_to_cuda(
                    example_batch[0], ignore=True, non_blocking=True
                )

                device = (
                    torch.device(f"cuda:{self._rank}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                example_images = _move_to_device(example_images, device)


                model_for_profile = copy.deepcopy(self.net).to(device)
                model_for_profile.eval()

                with torch.no_grad():

                    flops, _ = profile(model_for_profile, inputs=(example_images,), verbose=False)


                try:
                    if isinstance(example_images, (list, tuple)):
                        bs = len(example_images)
                    elif torch.is_tensor(example_images):
                        bs = int(example_images.shape[0])
                    else:
                        bs = 1
                    bs = max(1, bs)
                except Exception:
                    bs = 1
                stats["flops"] = float(flops) / float(bs)
                print(
                    f"[ModelStats] trainable_params={stats['trainable_params']}, "
                    f"total_params={stats['total_params']}, FLOPs={stats['flops']}"
                )
        except Exception as e:
            if self._rank == 0:
                print(f"[ModelStats] FLOPs profiling skipped or failed: {e}")


        stats["trainable_params_m"] = _to_m(stats.get("trainable_params"))
        stats["total_params_m"] = _to_m(stats.get("total_params"))
        stats["flops_g"] = _to_g(stats.get("flops"))
        return stats

    @torch.no_grad()
    def test_hico(self, dataloader, args=None, eval_mode='default'):
        """
        Test HICO-DET with different evaluation modes
        Args:
            eval_mode: 'default' or 'known_object'
                - 'default': Use correct_mat filtering (standard HICO-DET evaluation)
                - 'known_object': No correct_mat filtering (Known Object evaluation)
        """
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset
        while hasattr(dataset, 'dataset') and not hasattr(dataset, 'interaction_to_verb'):
            dataset = dataset.dataset


        visualizer = None
        visualization_data = []
        vis_predictions_list = []
        dump_vis_predictions = bool(
            args and (
                getattr(args, 'vis_dump_predictions', False)
                or args.vis_layout1
                or args.vis_layout2
                or getattr(args, 'vis_ablation_group', 0) > 0
            )
        )
        skip_vis_png = bool(args and getattr(args, 'vis_skip_png', False))
        if args and not skip_vis_png and (
            args.vis_layout1 or args.vis_layout2 or getattr(args, 'vis_ablation_group', 0) > 0
        ):
            visualizer = HOIVisualizer(args.vis_output_dir)
            print(f"Visualizer initialized. Output directory: {args.vis_output_dir}")
            print(f"Layout1: {args.vis_layout1}, Layout2: {args.vis_layout2}, AblationGroup: {getattr(args, 'vis_ablation_group', 0)}")
        elif dump_vis_predictions and skip_vis_png:
            os.makedirs(args.vis_output_dir, exist_ok=True)
            print(f"Prediction-only dump enabled. Output directory: {args.vis_output_dir}")
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)

        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        tgt_num_classes = 600


        correct_mat = None
        if eval_mode == 'default' and hasattr(dataset, 'correct_mat'):
            correct_mat = dataset.correct_mat
        elif eval_mode == 'default':

            try:
                correct_mat_path = 'hicodet/data/correct_mat.npy'
                if os.path.exists(correct_mat_path):
                    correct_mat = np.load(correct_mat_path)
                else:
                    print("Warning: correct_mat not found, using all interactions")
                    correct_mat = np.ones((117, 80), dtype=bool)
            except:
                print("Warning: Failed to load correct_mat, using all interactions")
                correct_mat = np.ones((117, 80), dtype=bool)
        elif eval_mode == 'known_object':

            correct_mat = np.ones((117, 80), dtype=bool)

        num_gt = dataset.anno_interaction if args.dataset == "hicodet" else None
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )

        gt_set = []
        pred_list = []

        vis_saved = 0
        sample_idx = 0
        subset_skipped = 0
        infer_times = []
        for batch in tqdm(dataloader):
            inputs = relocate_to_cuda(batch[0])
            targets = batch[-1]


            if getattr(args, 'eval_subset', 'all') != 'all':
                keep = [image_matches_eval_subset(t, args) for t in targets]
                if not any(keep):
                    subset_skipped += len(targets)
                    continue
                if not all(keep):

                    inputs = [inp for inp, k in zip(inputs, keep) if k]
                    targets = [t for t, k in zip(targets, keep) if k]
                    if len(inputs) == 0:
                        subset_skipped += sum(1 for k in keep if not k)
                        continue

            if getattr(args, 'profile_inference', False) and torch.cuda.is_available():
                torch.cuda.synchronize()
                t0 = time.time()

            outputs = net(inputs, targets)

            if getattr(args, 'profile_inference', False) and torch.cuda.is_available():
                torch.cuda.synchronize()
                infer_times.append((time.time() - t0) * 1000.0 / max(len(targets), 1))

            if outputs is None or len(outputs) == 0:
                if dump_vis_predictions or (
                    visualizer and (
                        getattr(args, 'vis_ablation_group', 0) > 0 or getattr(args, 'vis_all_images', False)
                    )
                ):
                    for target in targets:
                        vis_data = self._collect_visualization_data_empty(target)
                        if vis_data:
                            if dump_vis_predictions:
                                self._append_vis_prediction_record(
                                    vis_predictions_list, vis_data, args, png_basename=''
                                )
                            if visualizer and (
                                getattr(args, 'vis_ablation_group', 0) > 0 or getattr(args, 'vis_all_images', False)
                            ):
                                try:
                                    out_path = visualizer.save_per_image(vis_data, args, vis_saved)
                                    vis_saved += 1
                                    if dump_vis_predictions and out_path:
                                        vis_predictions_list[-1]['png_basename'] = os.path.basename(out_path)
                                    if vis_saved <= 5 or vis_saved % 100 == 0:
                                        print(f"Saved empty-detection vis to: {out_path}")
                                except Exception as e:
                                    print(f"Empty vis error: {e}")
                continue
            for output, target in zip(outputs, targets):
                output = relocate_to_cpu(output, ignore=True)
                sample_idx += 1


                if dump_vis_predictions or visualizer:
                    vis_data = self._collect_visualization_data(output, target, net)
                    if vis_data:
                        png_basename = ''
                        if dump_vis_predictions:
                            self._append_vis_prediction_record(
                                vis_predictions_list, vis_data, args, png_basename=''
                            )
                        if visualizer:

                            dump_this = getattr(args, 'vis_dump_per_image', False)
                            ablation_group = getattr(args, 'vis_ablation_group', 0)
                            vis_all = getattr(args, 'vis_all_images', False)

                            if dump_this or (ablation_group > 0) or vis_all:
                                stride = 1 if ((ablation_group > 0) or vis_all) else max(1, getattr(args, 'vis_stride', 10))
                                if sample_idx % stride != 0 and ablation_group == 0 and not vis_all:
                                    if len(visualization_data) < max(10, getattr(args, 'vis_topk', 5)):
                                        visualization_data.append(vis_data)
                                elif ablation_group == 0 and not vis_all:
                                    hoi_score = vis_data.get('confidence', 0.0)
                                    hi_t = getattr(args, 'vis_thresh', 0.5)
                                    lo_t = getattr(args, 'vis_low_thresh', 0.25)
                                    if not ((hoi_score >= hi_t) or (hoi_score <= lo_t)):
                                        if len(visualization_data) < max(10, getattr(args, 'vis_topk', 5)):
                                            visualization_data.append(vis_data)
                                    else:
                                        try:
                                            out_path = visualizer.save_per_image(vis_data, args, vis_saved)
                                            vis_saved += 1
                                            png_basename = os.path.basename(out_path) if out_path else ''
                                            if dump_vis_predictions and vis_predictions_list:
                                                vis_predictions_list[-1]['png_basename'] = png_basename
                                            if vis_saved % max(1, getattr(args, 'vis_flush_every', 50)) == 0:
                                                if ablation_group == 0 and (args.vis_layout1 or args.vis_layout2):
                                                    visualizer.visualize_model_outputs(visualization_data + [vis_data], args)
                                            if vis_saved <= 5 or vis_saved % 100 == 0:
                                                print(f"Saved per-image vis to: {out_path}")
                                        except Exception as e:
                                            print(f"Per-image visualization error: {e}")
                                else:
                                    try:
                                        out_path = visualizer.save_per_image(vis_data, args, vis_saved)
                                        vis_saved += 1
                                        png_basename = os.path.basename(out_path) if out_path else ''
                                        if dump_vis_predictions and vis_predictions_list:
                                            vis_predictions_list[-1]['png_basename'] = png_basename
                                        if vis_saved % max(1, getattr(args, 'vis_flush_every', 50)) == 0:
                                            if ablation_group == 0 and (args.vis_layout1 or args.vis_layout2):
                                                visualizer.visualize_model_outputs(visualization_data + [vis_data], args)
                                        if vis_saved <= 5 or vis_saved % 100 == 0:
                                            print(f"Saved per-image vis to: {out_path}")
                                    except Exception as e:
                                        print(f"Per-image visualization error: {e}")

                            if len(visualization_data) < max(10, getattr(args, 'vis_topk', 5)):
                                visualization_data.append(vis_data)
                                print(f"Collected visualization data {len(visualization_data)}: {vis_data['hoi_category']}")

                gt_set.append(target['hoi'])


                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                objects = output['objects']
                scores = output['scores']
                verbs = output['labels']

                if net.module.num_classes==117 or net.module.num_classes==407:
                    interactions = conversion[objects, verbs]
                    interactions = interactions.long()

                    if correct_mat is not None and eval_mode == 'default':
                        valid_mask = correct_mat[verbs.cpu().numpy(), objects.cpu().numpy()]
                        valid_mask = torch.from_numpy(valid_mask).to(verbs.device)
                        valid_indices = torch.nonzero(valid_mask).squeeze(1)
                        if len(valid_indices) > 0:
                            interactions = interactions[valid_indices]
                            scores = scores[valid_indices]
                            verbs = verbs[valid_indices]
                            objects = objects[valid_indices]
                            boxes_h = boxes_h[valid_indices]
                            boxes_o = boxes_o[valid_indices]
                        else:

                            continue
                else:
                    interactions = verbs


                gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])

                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()

                for hoi_idx in unique_hoi:

                    target_hoi = torch.as_tensor(target['hoi'], device=interactions.device)
                    gt_idx = torch.nonzero(target_hoi == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )


                results = (scores, interactions, labels)
                pred_list.append(results)

        gathered_pred_list = []
        for preds in ddp.all_gather(pred_list):
            gathered_pred_list.extend(preds)
        for pred in gathered_pred_list:
            meter.append(*pred)


        ap = meter.eval()

        stats = {
            'eval_subset': getattr(args, 'eval_subset', 'all'),
            'subset_skipped': subset_skipped,
            'num_evaluated': len(gt_set),
        }
        if infer_times:
            stats['infer_ms_per_image'] = float(sum(infer_times) / len(infer_times))
        self._last_eval_stats = stats
        if getattr(args, 'eval_subset', 'all') != 'all' and self._rank == 0:
            print(f"[EvalSubset={args.eval_subset}] evaluated={stats['num_evaluated']} skipped={subset_skipped}")
        if infer_times and self._rank == 0:
            print(f"[Profile] inference {stats['infer_ms_per_image']:.2f} ms/image")

        if dump_vis_predictions and self._rank == 0:
            self._save_vis_predictions(vis_predictions_list, args)

        return ap

    def _append_vis_prediction_record(self, records, vis_data, args, png_basename=''):
        """Append one top-1 prediction record for qualitative analysis."""
        filename = vis_data.get('filename', '')
        base_name = filename.replace('.jpg', '').replace('.png', '') if filename else ''
        pairs = vis_data.get('pairs') or []
        if pairs:

            verb = int(vis_data.get('verb', -1))
            obj = int(vis_data.get('object', -1))
            conf = float(vis_data.get('confidence', pairs[0].get('confidence', 0.0)))
            hoi_category = str(vis_data.get('hoi_category', pairs[0].get('hoi_category', '')))
        else:
            verb = int(vis_data.get('verb', -1))
            obj = int(vis_data.get('object', -1))
            conf = float(vis_data.get('confidence', 0.0))
            hoi_category = str(vis_data.get('hoi_category', ''))
        ablation_group = getattr(args, 'vis_ablation_group', 0)
        if not png_basename and base_name and ablation_group > 0:
            group_label = {2: 'baseline', 3: 'single', 4: 'two', 5: 'full'}.get(ablation_group, str(ablation_group))
            config = getattr(args, 'vis_ablation_config', '') or group_label
            png_basename = f"{base_name}_group{ablation_group}_{config}.png"
        records.append({
            'filename': filename,
            'png_basename': png_basename,
            'verb': verb,
            'object': obj,
            'confidence': conf,
            'hoi_category': hoi_category,
            'has_detection': bool(pairs),
        })

    def _save_vis_predictions(self, records, args):
        """Write per_image_predictions.json under vis_output_dir."""
        if not records:
            print("[VisPredictions] No records to save.")
            return
        out_dir = getattr(args, 'vis_output_dir', './visualizations')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'per_image_predictions.json')
        thresholds = {
            'interaction_success': getattr(args, 'vis_thresh', 0.5),
            'interaction_low_conf_failure': 0.35,
            'no_interaction_fp_failure': getattr(args, 'vis_thresh', 0.5),
        }
        payload = {
            'thresholds': thresholds,
            'count': len(records),
            'predictions': records,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[VisPredictions] Saved {len(records)} records to {out_path}")

    def _write_epoch_metrics(self, metrics: dict):
        """Append per-epoch metrics to a txt file under output_dir."""
        if self._rank != 0:
            return

        cache_dir = getattr(self, 'cache_dir', None)
        if cache_dir is None:
            cache_dir = getattr(self, '_cache_dir', './checkpoints')
        output_dir = getattr(self.args, 'output_dir', cache_dir)
        os.makedirs(output_dir, exist_ok=True)
        log_name = getattr(self.args, 'epoch_log_name', 'epoch_metrics.txt')
        log_path = os.path.join(output_dir, log_name)

        def _to_float(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return float(value)

        ordered_keys = [
            ('epoch', 'Epoch'),
            ('mAP', 'mAP'),
            ('rare', 'rare'),
            ('non-rare', 'non-rare'),
            ('mAP_known_object', 'Known'),
            ('rare_known_object', 'Known_rare'),
            ('non-rare_known_object', 'Known_non-rare'),
            ('full', 'Full'),
            ('unseen', 'Unseen'),
            ('seen', 'Seen'),

            ('training_time_min', 'Time(min/epoch)'),
            ('peak_gpu_mem_gb', 'GPU(GB)'),
            ('infer_ms_per_image', 'Infer(ms/img)'),

            ('trainable_params_m', 'Tr.Params(M)'),
            ('total_params_m', 'Tot.Params(M)'),
            ('flops_g', 'FLOPs(G)'),
        ]

        need_header = not os.path.exists(log_path)
        new_labels = [label for _, label in ordered_keys]
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                old_header = f.readline().strip()
            old_labels = [x.strip() for x in old_header.split('|')]
            if old_labels != new_labels:
                bak_path = log_path + '.bak'
                os.rename(log_path, bak_path)
                need_header = True
                print(f"[EpochMetrics] Extended header; previous log moved to {bak_path}")

        with open(log_path, 'a', encoding='utf-8') as f:
            if need_header:
                header = " | ".join(label for _, label in ordered_keys)
                f.write(header + '\n')

            values = []
            for key, label in ordered_keys:
                if key not in metrics:
                    values.append('')
                    continue
                try:
                    val = _to_float(metrics[key])
                    values.append(f"{val:.2f}" if key != 'epoch' else f"{int(val)}")
                except (ValueError, TypeError):
                    values.append(str(metrics[key]))
            f.write(" | ".join(values) + '\n')
    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction


        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = relocate_to_cuda(batch[0])
            output = net(inputs)


            if output is None or len(output) == 0:
                continue

            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = relocate_to_cpu(output[0], ignore=True)


            image_idx = dataset._idx[i]

            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]

            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct


            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1


            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]


            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num


        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )


        print(f"DEBUG: visualizer={visualizer is not None}, visualization_data={len(visualization_data) if visualization_data else 0}")
        if visualizer and visualization_data:
            print(f"Generating visualizations for {len(visualization_data)} samples...")
            visualizer.visualize_model_outputs(visualization_data, args)
        elif visualizer:
            print(f"No visualization data collected! (visualizer exists but data is empty)")
        elif visualization_data:
            print(f"Visualization data exists but no visualizer! (data count: {len(visualization_data)})")
        else:
            print("No visualizer and no visualization data.")

    def _collect_visualization_data(self, output, target, net):
        """Collect visualization data for HSE-CDR (TDAM attention branches)."""
        try:

            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']


            image = None
            for key in ['image', 'img', 'images']:
                if key in target and isinstance(target[key], (np.ndarray, torch.Tensor)):
                    image = target[key]
                    break


            if image is None:
                img_path = None
                for key in ['img_path', 'im_path', 'path', 'img_file', 'im_file', 'file_name', 'filename']:
                    if key in target and isinstance(target[key], str):
                        img_path = target[key]
                        break

                if img_path is not None and not os.path.exists(img_path):
                    base_root = getattr(self.args, 'data_root', './hicodet')

                    candidates = [
                        os.path.join(base_root, 'hico_20160224_det', 'images', 'test2015', img_path),
                        os.path.join(base_root, 'hico_20160224_det', 'images', 'train2015', img_path),
                        os.path.join('hicodet', 'hico_20160224_det', 'images', 'test2015', img_path),
                        os.path.join('hicodet', 'hico_20160224_det', 'images', 'train2015', img_path),
                    ]
                    for cand in candidates:
                        if os.path.exists(cand):
                            img_path = cand
                            break

                if img_path is not None and os.path.exists(img_path):
                    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if image is None:

                image = np.zeros((480, 640, 3), dtype=np.uint8)


            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
                if image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)


            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)


            try:
                H_img, W_img = image.shape[:2]
                if 'size' in output:
                    out_h, out_w = int(output['size'][0]), int(output['size'][1])
                    if out_h > 0 and out_w > 0 and (out_h != H_img or out_w != W_img):
                        sx = W_img / float(out_w)
                        sy = H_img / float(out_h)
                        boxes_h = boxes_h.clone()
                        boxes_o = boxes_o.clone()
                        boxes_h[:, 0::2] *= sx
                        boxes_h[:, 1::2] *= sy
                        boxes_o[:, 0::2] *= sx
                        boxes_o[:, 1::2] *= sy

                if boxes_h.max() <= 1.0 and boxes_o.max() <= 1.0:
                    boxes_h = boxes_h.clone()
                    boxes_o = boxes_o.clone()
                    boxes_h[:, 0::2] *= W_img
                    boxes_h[:, 1::2] *= H_img
                    boxes_o[:, 0::2] *= W_img
                    boxes_o[:, 1::2] *= H_img
            except Exception:
                pass


            hoi_scores = scores


            attention_data = {}


            if hasattr(net, 'module') and hasattr(net.module, 'last_attention_weights'):
                aw = net.module.last_attention_weights
                attention_data = aw
            elif hasattr(net, 'last_attention_weights'):
                aw = net.last_attention_weights
                attention_data = aw


            topk = min(len(scores), max(1, int(getattr(self.args, 'vis_topk', 5))))
            order = torch.argsort(hoi_scores, descending=True)[:topk]
            pairs = []
            for idx in order.tolist():
                hoi_category_i = self._determine_hoi_category(verbs[idx:idx+1], objects[idx:idx+1])
                pairs.append({
                    'human_box': boxes_h[idx].cpu().numpy().tolist(),
                    'object_box': boxes_o[idx].cpu().numpy().tolist(),
                    'confidence': float(hoi_scores[idx].item()),
                    'hoi_category': hoi_category_i,
                })


            hoi_category = pairs[0]['hoi_category'] if len(pairs) else 'unknown'
            top_idx = order[0].item() if len(order) > 0 else 0

            vis_data = {
                'hoi_category': hoi_category,
                'image': image,
                'human_box': boxes_h[top_idx].cpu().numpy().tolist() if len(boxes_h) > top_idx else [50, 50, 150, 150],
                'object_box': boxes_o[top_idx].cpu().numpy().tolist() if len(boxes_o) > top_idx else [200, 200, 300, 300],
                'confidence': float(hoi_scores[top_idx].item()) if len(hoi_scores) > top_idx else 0.5,
                'verb': verbs[top_idx].item() if len(verbs) > top_idx else 0,
                'object': objects[top_idx].item() if len(objects) > top_idx else 0,
                'pairs': pairs,
                'filename': target.get('filename', ''),
                **attention_data
            }

            return vis_data

        except Exception as e:
            print(f"Error collecting visualization data: {e}")
            return None

    def _collect_visualization_data_empty(self, target):
        """无检测时收集最小可视化数据（仅图像+filename）"""
        try:
            image = None
            img_path = target.get('filename', '')
            base_root = getattr(self.args, 'data_root', './hicodet')
            candidates = [
                os.path.join(base_root, 'hico_20160224_det', 'images', 'test2015', img_path),
                os.path.join(base_root, 'hico_20160224_det', 'images', 'train2015', img_path),
                os.path.join('hicodet', 'hico_20160224_det', 'images', 'test2015', img_path),
                os.path.join('hicodet', 'hico_20160224_det', 'images', 'train2015', img_path),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    img_bgr = cv2.imread(cand, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    break
            if image is None:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            return {
                'hoi_category': 'no_detection',
                'image': image,
                'human_box': [0, 0, 1, 1],
                'object_box': [0, 0, 1, 1],
                'confidence': 0.0,
                'pairs': [],
                'filename': img_path,
            }
        except Exception as e:
            print(f"Error in _collect_visualization_data_empty: {e}")
            return None

    def _determine_hoi_category(self, verbs, objects):
        """Determine HOI category from verb and object predictions"""

        try:
            from utils.hico_list import hico_verbs, hico_objects
        except Exception:
            hico_verbs, hico_objects = None, None

        if len(verbs) > 0 and len(objects) > 0:
            verb_id = int(verbs[0].item())
            obj_id = int(objects[0].item())

            if hico_verbs is not None and 0 <= verb_id < len(hico_verbs):
                verb = hico_verbs[verb_id]
            else:
                verb = f'verb_{verb_id}'

            if hico_objects is not None and 0 <= obj_id < len(hico_objects):
                obj = hico_objects[obj_id]
            else:
                obj = f'obj_{obj_id}'

            return f"{verb} {obj}"
        return "unknown"

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = relocate_to_cuda(batch[0])
            output = net(inputs)


            if output is None or len(output) == 0:
                continue

            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = relocate_to_cpu(output[0], ignore=True)


            image_id = dataset.image_id(i)

            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']

            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        return all_results
