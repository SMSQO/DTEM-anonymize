#!/usr/bin/env python3 

from typing import Dict
from ..general import load_yaml_cfg, load_contributor_index
import pickle
import numpy as np
from tqdm import tqdm
import torch 

cfg = load_yaml_cfg()['alpha']

emb_cfg = cfg['embedding']  # embedding configurations 

def main(): 
 
    embs = []
    for emb_type in {'repo', 'issue', 'api'}:
        with open(emb_cfg[f'contributor_{emb_type}_embedding'], 'rb') as fp: 
            embs.append(pickle.load(fp))

    contr_idx = load_contributor_index()  # contributor indices

    ret = [None] * len(contr_idx)
    for name, idx in tqdm(contr_idx.items()): 
        ret[idx] = np.concatenate([it[name] for it in embs])

    ret = np.array(ret)

    ret = torch.from_numpy(ret)
    torch.save(ret, emb_cfg['contributor_merged_embedding'])  # 394474 x 580


main()