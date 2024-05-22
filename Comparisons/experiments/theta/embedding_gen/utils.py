
import numpy as np
from typing import *
import torch
from tqdm import tqdm

def kv_emb_to_tensor(
        emb_dict: Dict[str, np.ndarray], 
        idx: Dict[str, int], 
        device=torch.device('cpu')
) -> np.ndarray:
 
    emb_size = next(iter(emb_dict.values())).shape[0]
    cnt = len(idx)

    ret = np.zeros((cnt, emb_size), dtype=np.float32)
    missing_emb = 0

    for repo_name in tqdm(idx):
        if repo_name not in emb_dict:
            missing_emb += 1
            continue
        ret[idx[repo_name]] = emb_dict[repo_name]

    print(f'Missing {missing_emb} embeddings.')
    return ret