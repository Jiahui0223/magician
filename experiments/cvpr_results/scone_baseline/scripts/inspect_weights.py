"""Phase A: inspect the official SCONE weights and compare them with the MACARONS model files (CPU only).

Usage: env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python -I inspect_weights.py
"""
import torch

W = '/bsuscratch/jiahuizhang/projects/occupancy/SCONE_official/weights_gdrive'
M = '/bsuscratch/jiahuizhang/projects/occupancy/MACARONS/weights/macarons'


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def strip(sd):
    return {k.replace('module.', '', 1): v for k, v in sd.items()}


occ = load(f'{W}/occupancy/pretrained_scone_occ.pth')
vis = load(f'{W}/coverage_gain/pretrained_scone_vis.pth')
for name, d in [('pretrained_scone_occ', occ), ('pretrained_scone_vis', vis)]:
    print(name, 'epoch', d['epoch'], 'loss', d['loss'], 'n_tensors', len(d['model_state_dict']),
          'n_params', sum(v.numel() for v in d['model_state_dict'].values()))
refs = {'occupancy': strip(occ['model_state_dict']), 'visibility': strip(vis['model_state_dict'])}

for mname in ['pretrained_macarons.pth', 'initialized_macarons.pth', 'trained_macarons.pth']:
    m = load(f'{M}/{mname}')
    sd = m['model_state_dict'] if 'model_state_dict' in m else m
    sc = sd['scone']
    print('\n', mname, '| scone keys sample:', list(sc.keys())[:2], '| n', len(sc))
    for sub, ref in refs.items():
        msub = {k.split(sub + '.', 1)[1].replace('module.', '', 1): v for k, v in sc.items() if k.startswith(sub + '.') or ('.' + sub + '.') in k}
        common = set(msub) & set(ref)
        maxdiff = max((msub[k].float() - ref[k].float()).abs().max().item() for k in common) if common else None
        print(f'   {sub}: n_model={len(msub)} n_ref={len(ref)} common={len(common)} max|diff|={maxdiff}')
