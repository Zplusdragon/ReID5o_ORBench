import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []
    print(f'Using {args.lora_lr_factor} times learning rate for random init LoRAs and CrossFusionFormer')
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        if "lora" in key or 'mm_head' in key or 'cross' in key or 'classifier' in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lora_lr_factor # default 5.0

        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
