"""Download the community 8-bit checkpoint for local evaluation."""
import json
import os
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
from huggingface_hub import HfApi, snapshot_download

repo = 'mlx-community/Breeze-TTS-2-mlx-8bit'
revision_file = ROOT / 'checkpoints' / 'model-revision.json'
revision = json.loads(revision_file.read_text())['revision'] if revision_file.exists() else 'c6e4a2ff6ab9afba68b7853de802273ffe23fb49'
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
print(f'Downloading {repo} at {info.sha}', flush=True)
destination = ROOT / 'checkpoints' / 'breeze-8bit'
snapshot_download(repo, revision=info.sha, local_dir=destination,
                  ignore_patterns=['*.safetensors'], max_workers=4)
for item in info.siblings:
    if not item.rfilename.endswith('.safetensors'):
        continue
    target = destination / item.rfilename
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.lfs.sha256
    if target.exists() and hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() == expected:
        print(f'Already verified: {item.rfilename}', flush=True)
        continue
    partial = target.with_suffix('.partial')
    size = item.size
    block = 32 * 1024 * 1024
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    url = f'https://huggingface.co/{repo}/resolve/{info.sha}/{item.rfilename}'
    def fetch(start):
        end = min(start + block, size) - 1
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise RuntimeError('Server did not honor requested byte range')
            pos = start
            while data := response.read(1024 * 1024):
                if pos + len(data) > end + 1:
                    raise RuntimeError('Oversized range response')
                view = memoryview(data)
                while view:
                    written = os.pwrite(fd, view, pos)
                    pos += written
                    view = view[written:]
            if pos != end + 1:
                raise RuntimeError('Incomplete range response')
        return end - start + 1
    try:
        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, start) for start in range(0, size, block)]
            for future in as_completed(futures):
                count = future.result()
                done += count
                print(f'{item.rfilename}: {done / size:.0%} ({done / 1e9:.2f}/{size / 1e9:.2f} GB)', flush=True)
    finally:
        os.close(fd)
    with partial.open('rb') as file:
        actual = hashlib.file_digest(file, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError(f'Checksum mismatch for {item.rfilename}')
    partial.replace(target)
    print(f'SHA256 verified: {item.rfilename}', flush=True)
(ROOT / 'checkpoints' / 'model-revision.json').write_text(json.dumps({'repo': repo, 'revision': info.sha}, indent=2) + '\n')
print('Download complete.', flush=True)
