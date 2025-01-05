import torch

def get_optimizer(
    params_to_optimize,
    args,
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=1e-3,
    eps=1e-8,
):
    # Optimizer creation
    supported_optimizers = ["adam", "adamw"]
    if args.optimizer not in supported_optimizers:
        print(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"
        
    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        print(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = (
            bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW
        )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    return optimizer
