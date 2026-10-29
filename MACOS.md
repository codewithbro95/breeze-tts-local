# Apple Silicon local setup

This is a checkout of the official Breeze repository with a macOS arm64
compatibility layer. The original CUDA/PyTorch source is retained. Apple Silicon
entry points use MLX-Audio 0.6.3 and the tested community 8-bit conversion.
This is not the original PyTorch computation running on a Mac, nor an official
BreezeBlue Mac release. The CLI argument names and HTTP request formats are retained.

## Generate in Terminal

```bash
cd /path/to/breeze-tts-local
source .venv/bin/activate
python infer.py checkpoints/breeze-8bit \
  --text "Machine learning helps computers discover patterns in data and use those patterns to make predictions." \
  --seed 42 \
  --output outputs/my-test.wav
afplay outputs/my-test.wav
```

The default voice is exactly the original research-news-01 prompt:

> Mature male, 50s-60s, Japanese-accented English. Dignified, measured, stoic tone. Clear texture with slight age rasp. Deliberate pacing, significant pauses. Conveys wisdom, authority, and visionary caution.

`configs/apple.json` records the original settings. CLI defaults: seed 42, CFG 4,
S0, temperature 0.9, top-p 1, top-k 50, repetition penalty 1, 400 tokens, non-streaming
WAV output. `--instruction` and `--cfg-scale` override voice design as upstream.
A reference pair without `--instruction` performs pure cloning, without injecting
our narration prompt. CLI text should be short enough for the 400-token limit
(about 32 seconds); a warning reports hitting that limit.

The README's `../breeze-tts-2` is an example checkpoint path. Substitute
`checkpoints/breeze-8bit` to use the self-contained local model. Reference audio
and its exact transcript must be supplied by you, as in upstream examples.

## API (same command and form fields)

```bash
python -m breeze_infer.api checkpoints/breeze-8bit --host 127.0.0.1 --port 7860
```

In another terminal:

```bash
curl --fail http://127.0.0.1:7860/health
curl --fail http://127.0.0.1:7860/v1/model
curl --fail http://127.0.0.1:7860/v1/audio/speech \
  -F 'text=Progress begins with a question.' -F 'seed=42' \
  --output outputs/api.pcm
ffplay -f s16le -ar 24000 -ch_layout mono outputs/api.pcm
```

The streaming endpoint returns raw 24 kHz mono s16le PCM, as upstream. Batch
requests accept the same fields and return concatenated PCM with X-Segment-Bytes.
MLX executes batch items sequentially to bound memory; it does not provide CUDA's
parallel batch speedup. One inference request at a time; concurrent requests get
409. The API retains upstream's 1500-token server ceiling. Streaming codec decoding
can differ from the non-streaming CLI WAV. The MLX model digest includes the codec
and uses a different digest scheme from the CUDA implementation.

CUDA graph acceleration (`--fast-*`) is unavailable on Apple Silicon. Those flags
are accepted with an explicit warning and use MLX. NVIDIA Docker build commands
remain NVIDIA-only; they are not a Mac inference route. Hub IDs are not fetched
by inference: pass the local MLX directory. `--revision` is ignored for local files.

## Environment, files, and reproduction

The `.venv`, `.cache`, `checkpoints`, and `outputs` directories are project-local.
There are no symlinks or runtime dependencies on the experiment directory. Model
files were copied independently from its verified weights; deleting the experiment
later will not break this project. The environment uses the host Python 3.12
interpreter, as normal for a venv. No hosted inference API or network access is
required at generation time; Hugging Face and Transformers offline modes are set.

To recreate the environment and download the pinned model:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv --python 3.12 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r requirements-mlx.lock.txt
source .venv/bin/activate
PIP_CACHE_DIR="$PWD/.cache/pip" python -m pip install -r requirements.txt
python download_model.py
```

The lock file pins the installed environment. `requirements.txt` retains upstream
CUDA dependencies on other platforms and selects MLX dependencies on macOS arm64.
`checkpoints/model-revision.json` pins the conversion. Model LICENSE and NOTICE
remain in its directory. The experiment is intentionally untouched.

## License

Self-hosted model weights and outputs retain BreezeBlue's research/non-commercial
license. This checkout does not grant commercial production rights; those require
a separate written license. See the original README and model LICENSE.
