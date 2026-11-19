#!/usr/bin/env python3
"""Convert the public HexHex pretrained checkpoint into a static ONNX model."""

import argparse
import configparser
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from urllib.request import urlopen

import onnx
import torch


UPSTREAM = "https://raw.githubusercontent.com/harbecke/HexHex/90d6cb1c7a983a1cfed907726c5964a22d663dd5"
MODEL_SOURCE = "hexhex/model/hexconvolution.py"
MODEL_SOURCE_SHA256 = "248d5b805001d626d377593a672d37321a3bf49e9752ace4020de10547c2a61d"
CHECKPOINT = "models/11_2w4_2000.pt"
CHECKPOINT_SHA256 = "bd0e4e3526ed5c241f02b8bb80b3999f85fe438c8ab763e34b4bb10ec507f46e"


def read_verified(path, expected_hash, upstream_dir):
    if upstream_dir is None:
        with urlopen(f"{UPSTREAM}/{path}", timeout=45) as response:
            data = response.read()
    else:
        data = (upstream_dir / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, help="Use a local checkout of the pinned HexHex revision.")
    parser.add_argument("--output", type=Path, default=Path("hex-ultra.onnx"))
    args = parser.parse_args()

    source = read_verified(MODEL_SOURCE, MODEL_SOURCE_SHA256, args.upstream_dir)
    checkpoint_bytes = read_verified(CHECKPOINT, CHECKPOINT_SHA256, args.upstream_dir)
    network = ModuleType("hexconvolution")
    exec(compile(source, MODEL_SOURCE, "exec"), network.__dict__)

    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([configparser.SectionProxy])
    with TemporaryDirectory() as scratch:
        checkpoint_path = Path(scratch) / "checkpoint.pt"
        checkpoint_path.write_bytes(checkpoint_bytes)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    board_size = config.getint("board_size")
    model = network.Conv(
        board_size, config.getint("layers"), config.getint("intermediate_channels"),
        config.getint("reach"), export_mode=False,
    )
    if not config.getboolean("switch_model"):
        model = network.NoSwitchWrapperModel(model)
    if config.getboolean("rotation_model"):
        model = network.RotationWrapperModel(model, export_mode=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = torch.zeros(1, 2, board_size + 2, board_size + 2)
    sample[0, 0, [0, -1], 1:-1] = 1
    sample[0, 1, 1:-1, [0, -1]] = 1
    torch.onnx.export(
        model, (sample,), args.output,
        input_names=["board"], output_names=["logits"], opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(args.output, full_check=True)


if __name__ == "__main__":
    main()
