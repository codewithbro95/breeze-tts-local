"""Compatibility for the upstream HTTP contract on Apple Silicon/MLX."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
import anyio
from .apple import SETTINGS, check_fast, generate, load_model, resolve_model


def pcm(audio):
    return (np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes()


def create_app(model):
    app = FastAPI(title='Breeze TTS API (Apple Silicon compatibility)')
    lock = threading.Lock()
    path = resolve_model(model)
    executor = ThreadPoolExecutor(max_workers=1)
    runtime = executor.submit(load_model, path).result()
    digest = hashlib.sha256()
    for file in sorted(path.rglob('*')):
        if file.is_file() and (file.suffix == '.safetensors' or file.name == 'config.json'):
            digest.update(str(file.relative_to(path)).encode())
            with file.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
    headers = {'X-Sample-Rate': '24000', 'X-Sample-Format': 's16le', 'Cache-Control': 'no-store'}

    @app.get('/health')
    def health():
        return {'status': 'ok', 'sample_rate': 24000}

    @app.get('/v1/model')
    def info():
        return dict(frame_rate=12.5, model_digest=digest.hexdigest(), max_new_tokens=1500, max_batch_texts=128)

    async def prepare(texts, instruction, cfg_scale, ref_audio, ref_text, seed, limit):
        if not math.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(400, 'cfg_scale must be greater than 0')
        if not isinstance(texts, list) or not 1 <= len(texts) <= 128 or not all(isinstance(t,str) and t.strip() for t in texts):
            raise HTTPException(400, 'Provide 1–128 nonempty texts')
        has_ref = ref_audio is not None and bool(ref_audio.filename)
        if has_ref != bool(ref_text.strip()):
            raise HTTPException(400, 'ref_audio and ref_text must be provided together')
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'An inference request is already running')
        temporary = None
        released = False
        def cleanup():
            nonlocal released
            if not released:
                released = True
                if temporary:
                    temporary.unlink(missing_ok=True)
                lock.release()
        try:
            if has_ref:
                payload = await ref_audio.read()
                if not payload:
                    raise HTTPException(400, 'Empty reference audio')
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temporary = Path(f.name)
                    f.write(payload)
            kwargs = dict(instruction=instruction, ref_audio=temporary, ref_text=ref_text or None,
                          seed=seed, cfg_scale=cfg_scale, max_tokens=limit)
            return kwargs, cleanup
        except BaseException:
            cleanup()
            raise

    class OwnedStream(StreamingResponse):
        def __init__(self, body, cleanup):
            self._stream, self.cleanup = body, cleanup
            super().__init__(body, media_type='audio/pcm', headers=headers)
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                with anyio.CancelScope(shield=True):
                    await self._stream.aclose()
                    self.cleanup()

    @app.post('/v1/audio/speech')
    async def speech(text: str = Form(...), instruction: str | None = Form(None),
                     cfg_scale: float = Form(4), ref_audio: UploadFile | None = File(None),
                     ref_text: str = Form(''), seed: int = Form(42)):
        kwargs, cleanup = await prepare([text], instruction, cfg_scale, ref_audio, ref_text, seed, 1500)
        async def body():
            iterator = generate(runtime, text, stream=True, **kwargs)
            sentinel = object()
            loop = asyncio.get_running_loop()
            def next_pcm():
                result = next(iterator, sentinel)
                return sentinel if result is sentinel else pcm(result[0])
            try:
                while True:
                    chunk = await loop.run_in_executor(executor, next_pcm)
                    if chunk is sentinel:
                        break
                    yield chunk
            finally:
                with anyio.CancelScope(shield=True):
                    await loop.run_in_executor(executor, iterator.close)
                cleanup()
        try:
            return OwnedStream(body(), cleanup)
        except BaseException:
            cleanup()
            raise

    @app.post('/v1/audio/speech/batch')
    async def batch(texts: str = Form(...), instruction: str | None = Form(None),
                    cfg_scale: float = Form(4), ref_audio: UploadFile | None = File(None),
                    ref_text: str = Form(''), seed: int = Form(42), max_new_tokens: int = Form(1500)):
        try:
            texts = json.loads(texts)
        except ValueError:
            raise HTTPException(400, 'texts must be a JSON array')
        kwargs, cleanup = await prepare(texts, instruction, cfg_scale, ref_audio, ref_text, seed, max(1,min(max_new_tokens,1500)))
        def work():
            try:
                # Sequential MLX execution bounds memory on 16 GB Macs; wire format is unchanged.
                return [b''.join(pcm(a) for a, _, _ in generate(runtime,t,**kwargs)) for t in texts]
            finally:
                cleanup()
        segments = await run_in_threadpool(lambda: executor.submit(work).result())
        return Response(b''.join(segments), media_type='application/octet-stream',
                        headers={**headers, 'X-Segment-Bytes': ','.join(str(len(s)) for s in segments)})
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--revision', help='Ignored for local checkpoints')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    for flag in ('all','text-encoder','backbone-prefill','backbone-decode','depth-decoder','codec'):
        parser.add_argument('--fast-'+flag, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    check_fast(args)
    import uvicorn
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)
