"""Run against a local API already started on port 7861; writes QA artifacts."""
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf

base='http://127.0.0.1:7861'
reference='Progress begins with a question. Today, researchers are exploring new ways to turn complex ideas into practical tools. The challenge is to move carefully, test the evidence, and build something worthy of trust.'
def request(name, endpoint, fields):
    cmd=['curl','--fail','--silent','--show-error',base+endpoint,'--dump-header',f'outputs/{name}.headers','--output',f'outputs/{name}.pcm']
    for key,value in fields.items(): cmd += ['-F',f'{key}={value}']
    subprocess.run(cmd,check=True)
    path=Path(f'outputs/{name}.pcm')
    data=path.read_bytes()
    audio=np.frombuffer(data,dtype='<i2')
    assert len(audio)>0 and np.any(audio), name
    assert '200 OK' in path.with_suffix('.headers').read_text()
    if name=='batch':
        header=path.with_suffix('.headers').read_text()
        lengths=next(line.split(':',1)[1] for line in header.splitlines() if line.lower().startswith('x-segment-bytes:'))
        assert sum(map(int,lengths.split(',')))==len(data)
        assert len(lengths.split(','))==2
    sf.write(path.with_suffix('.wav'),audio,24000,subtype='PCM_16')
    print(name,'passed',len(audio)/24000,'seconds',flush=True)
request('stream','/v1/audio/speech',{'text':'Welcome to the news.','seed':'42'})
request('batch','/v1/audio/speech/batch',{'texts':json.dumps(['Hello.','Thank you.']),'instruction':'A calm, clear narrator.','cfg_scale':'4','seed':'42','max_new_tokens':'80'})
fields={'text':'The evidence is clear.','ref_audio':'@outputs/research-news-01.wav','ref_text':reference,'seed':'42'}
request('clone','/v1/audio/speech',fields)
request('direction','/v1/audio/speech',{**fields,'instruction':'Speak slowly with a restrained, serious tone.','cfg_scale':'4'})
print('All Apple HTTP synthesis modes passed.',flush=True)
