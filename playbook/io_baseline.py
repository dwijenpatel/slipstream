#!/usr/bin/env python3
"""Invoked by fill_table.sh --io-baseline. No cache purge or model prewarming.
Alternates the same binary with/without F_NOCACHE; never assumes it bypasses
existing pages. Full stdout, stderr, per-window physical reads and drift persist.
"""
import argparse, csv, ctypes, datetime, hashlib, json, mmap, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--slots', default='16,64')
parser.add_argument('--contexts', default='3k')
parser.add_argument('--passes', type=int, default=2)
parser.add_argument('--max-new', type=int, default=1024)
parser.add_argument('--uncached-only', action='store_true',
                    help='run only cache-disabled arms; does not itself clear existing pages')
a = parser.parse_args()
slots = [int(s) for s in a.slots.split(',')]
contexts = a.contexts.split(',')
if a.passes < 2 or a.max_new < 512 or any(s not in (16,32,64,96,128,192) for s in slots):
    parser.error('requires >=2 passes, >=512 generated tokens, supported slot counts')
for c in contexts:
    if c not in ('1k','3k','12k','24k'): parser.error('unknown context')
model = pathlib.Path(os.environ.get('MODEL_GTURBO', pathlib.Path.home()/'models/qwen36.gturbo'))
binary = ROOT/'.build/release/slipstream'
if not (model/'manifest.json').is_file() or not binary.is_file():
    sys.exit('completed model and release binary required')
out = ROOT/'bench-results'/('io-baseline-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
out.mkdir(parents=True)

def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return {'command':args, 'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}

def idle():
    p = command(['pgrep','-fl','slipstream|TurboFieldfare|llama-server|mlx_lm'])
    if p['returncode'] != 1: raise RuntimeError('cannot establish idle model state: '+str(p))

def residency():
    # Map addresses only; never touch or fault expert data into memory.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_longlong]
    libc.mincore.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    page = os.sysconf('SC_PAGESIZE')
    result = []
    for path in sorted((model/'packed_experts').glob('*.bin')):
        size = path.stat().st_size
        fd = os.open(path,os.O_RDONLY)
        try:
            address = libc.mmap(None,size,mmap.PROT_READ,mmap.MAP_SHARED,fd,0)
            if address == ctypes.c_void_p(-1).value: raise OSError(ctypes.get_errno(),'mmap')
            try:
                pages = (size+page-1)//page
                vec = (ctypes.c_ubyte*pages)()
                if libc.mincore(address,size,vec): raise OSError(ctypes.get_errno(),'mincore')
                result.append({'file':path.name,'pages':pages,'resident_pages':sum(bool(v&1) for v in vec)})
            finally: libc.munmap(address,size)
        finally: os.close(fd)
    return {'page_bytes':page,'files':result,'resident_bytes':sum(r['resident_pages'] for r in result)*page}

idle()
meta = {k:command(v) for k,v in {
    'commit':['git','rev-parse','HEAD'], 'status':['git','status','--short'],
    'macos':['sw_vers'], 'swift':['swift','--version'],
    'hardware':['system_profiler','SPHardwareDataType','SPDisplaysDataType'],
    'memory':['memory_pressure','-Q'], 'wired':['sysctl','iogpu.wired_limit_mb']}.items()}
# Hardware output may include identifying machine fields: keep only specification lines.
meta['hardware']['stdout']='\n'.join(l for l in meta['hardware']['stdout'].splitlines()
    if any(k in l for k in ('Model Name:','Model Identifier:','Chip:','Total Number of Cores:','Memory:','Chipset Model:','VRAM','Metal Support:')))
meta.update(arguments=vars(a), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            cache_policy='no explicit prewarm or purge; alternating default/nocache; full SHA verification retained')
(out/'meta.json').write_text(json.dumps(meta,indent=2))
(out/'source.diff').write_text(command(['git','diff'])['stdout'])
for p in ('playbook/io_baseline.py','playbook/expert_io_probe.c','Sources/TurboFieldfareCLI/IOBaselineTelemetry.swift'):
    (out/pathlib.Path(p).name).write_bytes((ROOT/p).read_bytes())
fields=['arm','context','pass','slots','prefill_s','decode_tok_s','peak_gib','output_sha256','expert_bytes','disk_read_bytes','disk_to_expert_ratio','expert_resident_before_bytes','expert_resident_after_bytes']
rows=[]
arms=[(c,s,n) for c in contexts for s in slots for n in ((1,) if a.uncached_only else (0,1))]
print(out,flush=True)

def run(c,s,n,tag):
    idle()
    name=f'slots{s}-'+('nocache' if n else 'default')
    stem=f'{name}.{c}.{tag}'
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('TURBO_FIELDFARE_'): del env[key]
    env.update(TURBO_FIELDFARE_EXPERT_NOCACHE=str(n),TURBO_FIELDFARE_IO_BASELINE='1',TURBO_FIELDFARE_PHASES='1')
    cmd=['/usr/bin/time','-l',str(binary),'--model',str(model),'--messages-file',str(ROOT/f'playbook/prompts/ctx-{c}.json'),'--max-new',str(a.max_new),'--max-context','32768','--temperature','0','--seed','20260723','--expert-cache-slots',str(s)]
    (out/f'{stem}.command.json').write_text(json.dumps({'argv':cmd,'environment':{k:v for k,v in env.items() if k.startswith('TURBO_FIELDFARE_')}},indent=2))
    before={k:command(v) for k,v in {'vm':['vm_stat'],'swap':['sysctl','vm.swapusage'],'memory':['memory_pressure','-Q']}.items()}
    before['expert_residency'] = residency()
    print('running',stem,flush=True)
    with (out/f'{stem}.stdout').open('wb') as stdout, (out/f'{stem}.log').open('wb') as stderr:
        p=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr)
    after={k:command(v) for k,v in {'vm':['vm_stat'],'swap':['sysctl','vm.swapusage'],'memory':['memory_pressure','-Q']}.items()}
    after['expert_residency'] = residency()
    (out/f'{stem}.system.json').write_text(json.dumps({'before':before,'after':after},indent=2))
    if p.returncode: raise RuntimeError(f'{stem} failed: inspect log')
    log=(out/f'{stem}.log').read_text()
    footer=re.search(r'\[stop=[^\]]+\]',log)
    if not footer: raise RuntimeError('missing timing footer')
    foot=footer.group()
    pre=float(re.search(r'prefill=\d+tok/([0-9.]+)s',foot)[1])
    dec=float(re.search(r'tok/s=([0-9.]+)',foot)[1])
    windows=re.findall(r'\[io-window .*?expert_bytes=(\d+) disk_read_bytes=(\d+)',log)
    if not windows or '[io-window error=' in log: raise RuntimeError('physical I/O telemetry unavailable')
    logical=sum(int(x) for x,y in windows); disk=sum(int(y) for x,y in windows)
    peak=re.search(r'(\d+)\s+peak memory footprint',log)
    sha=hashlib.sha256((out/f'{stem}.stdout').read_bytes()).hexdigest()
    row=dict(zip(fields,[name,c,tag,s,pre,dec,int(peak[1])/2**30 if peak else '',sha,logical,disk,disk/logical if logical else '',before['expert_residency']['resident_bytes'],after['expert_residency']['resident_bytes']]))
    rows.append(row)
    with (out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(row,flush=True)

for p in range(1,a.passes+1):
    for c,s,n in arms: run(c,s,n,f'pass{p}')
# Recheck each arm; cache regimes can drift differently.
for c,s,n in arms: run(c,s,n,'drift')
checks=[]
for c,s,n in arms:
    name=f'slots{s}-'+('nocache' if n else 'default')
    b=next(r for r in rows if r['arm']==name and r['context']==c and r['pass']==f'pass{a.passes}')
    d=next(r for r in rows if r['arm']==name and r['context']==c and r['pass']=='drift')
    checks.append({'arm':name,'context':c,'decode_drift_percent':100*(d['decode_tok_s']/b['decode_tok_s']-1),'prefill_drift_percent':100*(d['prefill_s']/b['prefill_s']-1)})
parity=all(len({r['output_sha256'] for r in rows if r['context']==c})==1 for c in contexts)
summary={'drift':checks,'byte_identical_outputs':parity,'drift_pass':all(abs(x[k])<=5 for x in checks for k in ('decode_drift_percent','prefill_drift_percent')),
         'uncached_evidence_pass':a.uncached_only and all(r['expert_resident_before_bytes']==0 and r['expert_resident_after_bytes']==0 and r['disk_to_expert_ratio']>=0.99 for r in rows),
         'warning':'F_NOCACHE can hit existing pages. A disk/expert ratio substantially below one rejects the strict uncached claim. Process disk reads can also include unrelated reads.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
