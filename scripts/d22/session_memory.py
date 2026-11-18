import json, re, sys, glob, os
def load(d):
    sys_ = []
    for l in open(os.path.join(d, 'sys.log')):
        kv = dict(re.findall(r'(\w+)=(-?[\d.]+)', l))
        if 'epoch' in kv: sys_.append({k: float(v) for k, v in kv.items()})
    turns = [json.loads(l) for l in open(os.path.join(d, 'turns.jsonl'))]
    turns = [t for t in turns if 't_send' in t]
    return sys_, turns
def win(sys_, a, b, key, f):
    v = [s[key] for s in sys_ if a <= s['epoch'] <= b and key in s]
    return f(v) if v else None
for d in sorted(sys.argv[1:]):
    s, ts = load(d)
    print('==', os.path.basename(d), 'samples', len(s))
    print(' fill_k | pre: avail_min gtt_max cache_max | dec: avail_min gtt_max cache | new_tok')
    for t in ts:
        u = t.get('usage') or {}
        fill = (u.get('prompt_tokens') or 0) + (u.get('completion_tokens') or 0)
        tf = t.get('t_first') or t['t_end']
        p = (win(s, t['t_send'], tf, 'avail_mb', min), win(s, t['t_send'], tf, 'gtt_mb', max), win(s, t['t_send'], tf, 'cache_mb', max))
        q = (win(s, tf, t['t_end'], 'avail_mb', min), win(s, tf, t['t_end'], 'gtt_mb', max), win(s, tf, t['t_end'], 'cache_mb', max))
        new = (u.get('prompt_tokens') or 0) - ((u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0)
        print(f" {fill/1000:6.1f} | {p[0]} {p[1]} {p[2]} | {q[0]} {q[1]} {q[2]} | {new}")
