"""AR V1 training entrypoint: same W1 data/trainer/flags as Diffusion.

Equivalent to: python train_diffusion_pretrain.py --generator-type ar_v1 ...
"""
import sys
from train_diffusion_pretrain import main

if __name__ == "__main__":
    sys.argv.insert(1, "--generator-type")
    sys.argv.insert(2, "ar_v1")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(("--output-dir", "outputs/ar_v1_pretrain"))
    raise SystemExit(main())
