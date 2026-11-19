Play against my Hex agent

Serve this directory with any static web server. For a local game, open a
terminal in this directory and run:

    python3 -m http.server 8000 --bind 127.0.0.1

Then visit http://127.0.0.1:8000 in a current browser. The app loads its
worker, ONNX model and WebAssembly over HTTP. Once the files are in place,
the game can run offline and needs no account, CDN or backend.

Choose who starts, then select Start. Red always opens and connects
top to bottom. Blue connects left to right. Select New Game to return to
setup, then Start to begin the next match.
On turn two, Blue can select the highlighted red opening stone to swap
colours. The stone stays in place, and the original opener moves next as Blue.

The optional heatmap shows legal-cell policy logits for the agent's
colour, scaled from the lowest to the highest score on the current
board. Yellow means the agent used a rule for its last move. If the
agent starts, the heatmap previews its opening cell during setup.

Agent: COMP34111 Group 13.
Pretrained policy: HexHex by Simon Buchholz, David Harbecke and Pascal
Van Cleeff. The original 11_2w4_2000.pt checkpoint is byte-for-byte
identical to the Group 13 checkpoint, with SHA-256:
bd0e4e3526ed5c241f02b8bb80b3999f85fe438c8ab763e34b4bb10ec507f46e

The pinned public checkpoint, source and training tooling are at:
https://github.com/harbecke/HexHex/tree/90d6cb1c7a983a1cfed907726c5964a22d663dd5
See notices/ for the model export recipe and third-party notices.
