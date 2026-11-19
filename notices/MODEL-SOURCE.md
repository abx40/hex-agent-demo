# Hex Ultra model source

COMP34111 Group 13’s `hex_ultra.pt` is byte-for-byte identical to the pretrained HexHex checkpoint `11_2w4_2000.pt`. HexHex is by Simon Buchholz, David Harbecke, and Pascal Van Cleeff. The checkpoint and network code come from the pinned upstream revision below. The accompanying `HexHex-GPL-3.0.txt` contains the upstream license.

- [HexHex upstream revision](https://github.com/harbecke/HexHex/tree/90d6cb1c7a983a1cfed907726c5964a22d663dd5)
- [Pretrained checkpoint](https://github.com/harbecke/HexHex/blob/90d6cb1c7a983a1cfed907726c5964a22d663dd5/models/11_2w4_2000.pt)
- [Network implementation](https://github.com/harbecke/HexHex/blob/90d6cb1c7a983a1cfed907726c5964a22d663dd5/hexhex/model/hexconvolution.py)

From this directory, run these commands with Python 3.12.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install onnx
python reproduce-model.py --output hex-ultra.onnx
```

The script reads both public inputs from the pinned revision and verifies their SHA-256 digests before loading them. It exports the rotation wrapper and the occupied-cell penalty, with input `board` of shape `[1, 2, 13, 13]` and output `logits` of shape `[1, 121]`. A pinned local checkout can be supplied with `--upstream-dir`.

The demo runs inference with ONNX Runtime Web 1.31.1, licensed under MIT. The accompanying Microsoft license and third-party notices are from the matching ONNX Runtime release package. [ONNX Runtime Web 1.31.1](https://www.npmjs.com/package/onnxruntime-web/v/1.31.1) · [ONNX Runtime 1.31.1](https://pypi.org/project/onnxruntime/1.31.1/)
