"""
HSE-CDR training, evaluation and result caching for HICO-DET and V-COCO.
"""
import os
import socket
import torch
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
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

from models.hse_cdr import build_detector
from utils.args import get_args
from utils.eval_utils import (
    find_checkpoint, resolve_checkpoint, summarize_hico_ap,
    save_eval_json, parse_hyper_lambda_sweep,
)
from utils.ablation_registry import get_experiment

import sys as _sys, os as _os


_ROOT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_CANDIDATE_POCKET_DIRS = [
    _os.path.join(_ROOT_DIR, "pocket"),
    _os.path.join(_ROOT_DIR, "..", "pocket"),
]
for _p in _CANDIDATE_POCKET_DIRS:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

try:
    from engine import CustomisedDLE
except Exception:
    for _k in list(_sys.modules.keys()):
        if _k == 'pocket' or _k.startswith('pocket.'):
            del _sys.modules[_k]
    from engine import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")


def _pick_master_port(preferred=None):
    """Return a free TCP port; try preferred first, then OS-assigned."""
    if preferred is not None:
        try:
            port = int(preferred)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return str(port)
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return str(s.getsockname()[1])


def _apply_ablation_config(args, exp_id: str):
    """Override args from ablation registry entry."""
    if not exp_id:
        return args
    cfg = get_experiment(exp_id)
    for key, val in cfg.items():
        if key in ('exp_id', 'name', 'train', 'checkpoint', 'checkpoint_dirs', 'prefer_epoch'):
            continue
        attr = key.replace('-', '_')
        if hasattr(args, attr):
            setattr(args, attr, val)
    args.ablation_id = exp_id
    if cfg.get('output_dir'):
        args.output_dir = cfg['output_dir']
    if cfg.get('hyper_lambda_sweep'):
        args.hyper_lambda_sweep = cfg['hyper_lambda_sweep']
    return args


def _run_hico_eval(engine, trainset, args):
    """Run HICO eval and return metrics dict."""
    ap = engine.test_hico(engine.test_loader, args)
    num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
    print(
        f"The mAP is {ap.mean()*100:.2f},"
        f" rare: {ap[rare].mean()*100:.2f},"
        f" none-rare: {ap[non_rare].mean()*100:.2f},"
    )
    metrics = summarize_hico_ap(ap, args, trainset)
    if args.zs:
        print(
            f'>>> zero-shot setting({args.zs_type})',
            f"full mAP: {metrics['full']:.2f}",
            f"unseen: {metrics['unseen']:.2f}",
            f"seen: {metrics['seen']:.2f}",
        )
    extra = getattr(engine, '_last_eval_stats', {}) or {}
    metrics.update(extra)
    if hasattr(engine, 'model_stats') and engine.model_stats:
        for k in ('trainable_params', 'total_params', 'flops'):
            if engine.model_stats.get(k) is not None:
                metrics[k] = engine.model_stats[k]
        if engine.model_stats.get('trainable_params') is not None:
            metrics['trainable_params_m'] = engine.model_stats['trainable_params'] / 1e6
        if engine.model_stats.get('total_params') is not None:
            metrics['total_params_m'] = engine.model_stats['total_params'] / 1e6
        if engine.model_stats.get('flops') is not None:
            metrics['flops_g'] = engine.model_stats['flops'] / 1e9
    return metrics


def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )


    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)


    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'


        args.clip_visual_layers_vit = 12
        args.clip_visual_output_dim_vit = 512
        args.clip_visual_input_resolution_vit = 224
        args.clip_visual_width_vit = 768
        args.clip_visual_patch_size_vit = 16
        args.clip_text_transformer_width_vit = 512
        args.clip_text_transformer_heads_vit = 8
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

        args.clip_visual_layers_vit = 24
        args.clip_visual_output_dim_vit = 768
        args.clip_visual_input_resolution_vit = 336
        args.clip_visual_width_vit = 1024
        args.clip_visual_patch_size_vit = 14
        args.clip_text_transformer_width_vit = 768
        args.clip_text_transformer_heads_vit = 12


    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                           num_classes=args.num_classes, args=args)
    full_testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                          clip_model_name=args.clip_model_name, args=args)
    testset = full_testset
    if args.eval and getattr(args, 'eval_max_images', 0) > 0:
        n = min(args.eval_max_images, len(full_testset))
        testset = torch.utils.data.Subset(full_testset, range(n))
        if dist.get_rank() == 0:
            print(f'[EvalMaxImages] Using first {n} test images only')


    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        sampler=DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=rank
        )
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        sampler=torch.utils.data.distributed.DistributedSampler(
        testset, shuffle=False, drop_last=False
        )
    )

    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    object_to_target = trainset.dataset.object_class_to_target_class


    print('[INFO]: num_classes', args.num_classes)
    model = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit)

    if args.dataset == 'hicodet' and args.eval:
        model.object_class_to_target_class = full_testset.dataset.object_class_to_target_class

    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        model, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        model.eval()
        if args.dataset == 'vcoco':
            import eval_vcoco
            ret = engine.cache_vcoco(test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            ap = vcocoeval._do_eval(ret, ovr_thresh=0.5)
            print(ap)
            return


        lambdas = parse_hyper_lambda_sweep(getattr(args, 'hyper_lambda_sweep', ''))
        if lambdas:
            sweep_results = []
            for lam in lambdas:
                model.hyper_lambda = lam
                args.hyper_lambda = lam
                if rank == 0:
                    print(f"\n=== hyper_lambda={lam} ===")
                m = _run_hico_eval(engine, trainset, args)
                m['hyper_lambda'] = lam
                sweep_results.append(m)
            if rank == 0:
                out = args.eval_output_json or os.path.join(
                    args.output_dir, f'ablation_{args.ablation_id or "eval"}_lambda_sweep.json'
                )
                save_eval_json(out, {
                    'ablation_id': args.ablation_id,
                    'checkpoint': args.resume,
                    'sweep': sweep_results,
                })
                print(f"Saved lambda sweep to {out}")
            return

        metrics = _run_hico_eval(engine, trainset, args)
        if rank == 0:
            out = args.eval_output_json or os.path.join(
                args.output_dir, f'ablation_{args.ablation_id or "eval"}_results.json'
            )
            payload = {
                'ablation_id': args.ablation_id,
                'checkpoint': args.resume,
                'hyper_lambda': args.hyper_lambda,
                'eval_subset': args.eval_subset,
                **metrics,
            }
            save_eval_json(out, payload)
            print(f"Saved eval results to {out}")
        return

    for p in model.detector.parameters():
        p.requires_grad = False

    for n, p in model.clip_head.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'):
            p.requires_grad = True
        elif 'adaptermlp' in n or ("prompt_learner" in n and args.use_prompt):
            p.requires_grad = True
        elif 'visual_prompt' in n:
            p.requires_grad = True
        else: p.requires_grad = False


    try:
        engine.model_stats = engine._compute_model_stats()
    except Exception as _e:
        if rank == 0:
            print(f"[ModelStats] refresh after freeze failed: {_e}")

    param_dicts = [
        {
            "params": [p for n, p in model.clip_head.named_parameters()
                    if p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters()
                    if p.requires_grad and 'clip_head' not in n],
            "lr": args.lr_head,
        },
    ]

    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_vit,
        weight_decay=args.weight_decay
    )

    if args.lr_milestones is not None and len(args.lr_milestones) > 0:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=args.lr_milestones, gamma=args.lr_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_gamma)

    if args.resume:


        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    import json
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    engine(args.epochs)

if __name__ == '__main__':
    args = get_args()


    if getattr(args, 'ablation_id', ''):
        args = _apply_ablation_config(args, args.ablation_id)

    if getattr(args, 'log_epoch_costs', False) and not args.eval:
        args.profile_inference = True


    if args.eval and getattr(args, 'auto_resume_best', False) and not os.path.isfile(args.resume):
        if args.ablation_id:
            ckpt = resolve_checkpoint(get_experiment(args.ablation_id))
        else:
            ckpt = find_checkpoint(args.output_dir)
        if ckpt:
            args.resume = ckpt
            print(f"[AutoResume] Using checkpoint: {ckpt}")

    print(args)


    os.environ['WANDB_MODE'] = 'disabled'
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    print('WORLD_SIZE ' + str(os.environ.get("WORLD_SIZE",1)))

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")

    preferred_port = os.environ.get("MASTER_PORT") or (args.port if args.debug else None)
    if args.eval or args.debug:
        chosen = _pick_master_port(preferred_port)
        os.environ["MASTER_PORT"] = chosen
        if preferred_port and str(preferred_port) != chosen:
            print(f"[MasterPort] {preferred_port} busy, using {chosen}")
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12360"
    local_rank = int(os.environ.get("LOCAL_RANK",0))
    args.local_rank = local_rank
    if local_rank == 0:
        wandb.init(project='HSE-CDR', name=args.output_dir)

    args.world_size = int(os.environ.get("WORLD_SIZE",1))
    main(local_rank,args)
