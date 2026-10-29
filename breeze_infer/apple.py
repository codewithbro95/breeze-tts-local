"""Apple Silicon compatibility using the pinned MLX-Audio Breeze implementation."""
import json
import os
from pathlib import Path
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TMPDIR'] = str(ROOT / '.cache/tmp')
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
SETTINGS = json.loads((ROOT / 'configs/apple.json').read_text())


def check_fast(args):
    if any(value for key, value in vars(args).items() if key.startswith('fast_')):
        warnings.warn('CUDA graph flags are unavailable on Apple Silicon; using native MLX inference.')


def resolve_model(value):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    raise ValueError('Use a local MLX checkpoint directory, such as checkpoints/breeze-8bit. Offline mode does not download models.')


def load_model(path):
    import mlx.core as mx
    from mlx_audio.tts import load
    if not mx.metal.is_available():
        raise RuntimeError('Apple Metal GPU unavailable; no CPU fallback.')
    mx.set_default_device(mx.gpu)
    model = load(str(resolve_model(path)))
    mx.eval(model.parameters())
    return model


def generate(model, text, instruction=None, ref_audio=None, ref_text=None,
             seed=42, cfg_scale=4, max_tokens=None, stream=False):
    import mlx.core as mx
    import numpy as np
    if not text.strip():
        raise ValueError('Text must not be empty')
    if bool(ref_audio) != bool(ref_text and ref_text.strip()):
        raise ValueError('Reference audio and complete transcript must be supplied together')
    # A reference without instruction is pure cloning, as in the upstream CLI.
    if instruction is None and ref_audio is None:
        instruction = SETTINGS['instruction']
    mx.set_default_device(mx.gpu)
    for result in model.generate(
        text=text, voice='S0', instruct=instruction,
        ref_audio=str(ref_audio) if ref_audio else None, ref_text=ref_text,
        seed=seed, cfg_scale=cfg_scale,
        max_tokens=max_tokens or SETTINGS['max_tokens'], stream=stream,
        temperature=SETTINGS['temperature'], top_p=SETTINGS['top_p'],
        top_k=SETTINGS['top_k'], repetition_penalty=SETTINGS['repetition_penalty'],
    ):
        mx.eval(result.audio)
        audio = np.asarray(result.audio, dtype=np.float32)
        if not np.isfinite(audio).all():
            raise RuntimeError('Non-finite generated audio')
        yield audio, result.sample_rate, result.token_count


def infer(args):
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    check_fast(args)
    start = time.perf_counter()
    model = load_model(args.model)
    loaded = time.perf_counter()
    chunks = list(generate(model, args.text, args.instruction, args.ref_audio,
                           args.ref_text, args.seed, args.cfg_scale))
    audio = np.concatenate([chunk[0] for chunk in chunks]) if chunks else np.array([])
    if not len(audio) or np.sqrt(np.mean(audio ** 2)) < 0.0001:
        raise RuntimeError('No usable audio generated')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, chunks[0][1], subtype='PCM_16')
    if sum(chunk[2] for chunk in chunks) >= SETTINGS['max_tokens']:
        warnings.warn('Generation reached the configured token limit; check for truncation.')
    print(f'saved {args.output}; load={loaded-start:.2f}s; generation={time.perf_counter()-loaded:.2f}s; audio={len(audio)/chunks[0][1]:.2f}s; peak MLX={mx.get_peak_memory()/1e9:.2f} GB', flush=True)
