import argparse
import os

from main.training import train
from main.prediction import test

def main():
    ap = argparse.ArgumentParser("SLCAT")

    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument(
        "--ckpt", type=str, help="checkpoint for prediction (default: <model_dir>/best.ckpt)"
    )

    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue training in <model_dir> from latest.ckpt",
    )

    ap.add_argument(
        "--output_path", type=str, help="path for saving translation output"
    )
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config_path, resume=args.resume)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
