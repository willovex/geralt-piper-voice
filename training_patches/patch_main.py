"""
Patch piper_train/__main__.py for pytorch-lightning 2.x compatibility.

PL 2.x removed Trainer.add_argparse_args / Trainer.from_argparse_args, so the
CLI args have to be declared and the Trainer constructed manually.

Usage: python3 patch_main.py /path/to/piper/src/python/piper_train/__main__.py
"""
import sys

path = sys.argv[1]
src = open(path).read()

old_args = """    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)"""
new_args = """    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--default_root_dir", default=None)
    VitsModel.add_model_specific_args(parser)"""
src = src.replace(old_args, new_args)

old_trainer = "    trainer = Trainer.from_argparse_args(args)"
new_trainer = """    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=str(args.default_root_dir) if args.default_root_dir else None,
    )"""
src = src.replace(old_trainer, new_trainer)

src = src.replace("    trainer.fit(model)", "    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)")

# Enable TF32 matmul on Ampere/Blackwell GPUs (RTX 5090 etc.)
src = src.replace(
    "    torch.manual_seed(args.seed)",
    "    torch.manual_seed(args.seed)\n    torch.set_float32_matmul_precision('high')",
)

open(path, "w").write(src)
print("patched", path)
