# Local validation — 2026-10-29

Official source: https://github.com/breezeblue-ai/breeze-tts
Upstream commit: `6eba901d1845353802652c708a4740fdfc5baa75`.
Local compatibility changes remain visible in `git diff` and new files.

Hardware: Apple M1 / 16 GB. Python 3.12.8; MLX 0.32.4; MLX-Audio 0.6.3.
Exact dependency versions are in requirements-mlx.lock.txt.
Checkpoint: mlx-community/Breeze-TTS-2-mlx-8bit at
`c6e4a2ff6ab9afba68b7853de802273ffe23fb49`.

Both independent model copies match the experiment's previously verified SHA256:

- Main: `d7a82315bcc59c3057f13bb85100743c7cd40692e38145fb255d6c1f0bf447bc`
- Codec: `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`

## CLI reproduction

`python infer.py checkpoints/breeze-8bit --text "<original paragraph>" --output outputs/research-news-01.wav`

Passed. Result is byte-for-byte identical to the user's preferred experimental
research-news-01.wav, SHA256:
`4970572b4c5241804d06fa36e150c66a58c9b35bb959b1e72802b8d75d43636a`.

17.04 seconds, mono 24 kHz PCM16. Load 16.05 seconds; generation 68.78 seconds;
peak MLX memory 8.81 GB. Timings depend on system load.

## HTTP compatibility

Started through the documented module entry point, bound to 127.0.0.1:7861.
`tests/apple_smoke.py` exercised real inference (not mocks):

- Streaming PCM: passed, 2.16 seconds.
- Batch of two texts: passed, 1.68 seconds combined; segment byte counts validated.
- Reference voice clone: passed, 1.60 seconds.
- Reference voice direction with explicit instruction: passed, 2.24 seconds.
- Unpaired reference, empty batch, non-finite CFG: HTTP 400; health remained OK.
- pip requirements installation, pip check, CLI help, API help, git diff whitespace check: passed.

Generated PCM was nonempty and non-silent; converted listening WAVs are in outputs/.
The temporary test API server was stopped after validation. Start your own using
MACOS.md. The experimental project was not modified or deleted.

## Scope

The original CUDA test suite requires NVIDIA/PyTorch and was not run on this Mac.
CUDA graph and Docker acceleration do not run on Apple Silicon. The Mac layer
uses MLX; batch items run sequentially; local checkpoint paths are required.
See MACOS.md for these platform differences and the retained non-commercial
model license. This validation is not a commercial license or a load/stress test.
