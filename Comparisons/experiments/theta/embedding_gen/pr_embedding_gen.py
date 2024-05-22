#!/usr/bin/env python3

'''
@see also:
    * NodeFeatureInitializer/embed_pr_text.py
'''

import json
from tqdm import tqdm
import os
import pickle
from llama3_embedding import Llama3Embedder
from typing import *
import numpy as np 
import torch
from utils import kv_emb_to_tensor

FILE_PREFIX = './data/pr_embedding'

class PREmbedder(): 

    def __init__(self): 
        self.model = Llama3Embedder(device='cuda:1')

    def text_embedding(self): 
        '''
        SRC_FILE => DST_FILE. 
        LIST <| JSON { 'text': ...; ...} => LIST <| Tensor<4096>
        '''
        # Sorry. Almost totally coied from repo_embedding_gen.py

        src_file = f'{FILE_PREFIX}/text_src.txt' 
        dst_file = f'{FILE_PREFIX}/text.pkl'

        if os.path.exists(dst_file): 
            print('"text.pkl" embedding file found, chosen.')
            with open(dst_file, 'rb') as fp: 
                return pickle.load(fp)  # JSON <| {project: string, number: int, text: string, code: string[]}
                
        def pr_seq() -> Iterable[Dict]: 
            with open(src_file) as fp: 
                for line in fp: 
                    yield json.loads(line)

        repo_text_lst = list(pr_seq())
        repo_text_embeddings = {
            f"{it['project']}##{it['number']}": self.model(it['text']) 
            for it in tqdm(repo_text_lst)
        }
        with open(dst_file, 'wb') as fp: 
            pickle.dump(repo_text_embeddings, fp)

        print(len(repo_text_embeddings))
        return repo_text_embeddings  # len(*): 378415 

    def original_code_embedding(self) -> Dict[str, np.ndarray]:  # Array<768>
 
        with open(f'{FILE_PREFIX}/original_code.pkl', 'rb') as fp:   # TODO check file existence
            return pickle.load(fp)  # JSON <| { $repo_name: Array<768> }
    

    # warning: never tested        
    # Sorry again. copied from repo_embedding_gen.py
    # Deprecated this if the the method in repo_embedding_gen is changed. 
    def new_code_embedding(self) -> Dict[str, np.ndarray]: 
        src_file = f'data/code_snippet/code_snippet.pkl'
        ...
        # if not os.path.exists(src_file): 
        #     raise Exception(f'"{src_file}" not exists. generate new code_snippet embeddings first!')

        # with open(src_file, 'rb') as fp: 
        #     data = pickle.load(fp)  # JSON <| { $project: { $path: Vector<4096> } }

        # return {
        #     proj_k: np.average(list(proj_v.values()),axis=1)
        #     for proj_k, proj_v in data.items()
        # }

    def merged_embedding(
            self, 
            *,
            device=torch.device('cpu'),
            use_old_code_embedding: bool,
        ) -> torch.Tensor:
 

        text_emb = self.text_embedding()
        code_emb = self.original_code_embedding() if use_old_code_embedding else self.new_code_embedding()

        def load_pr_idx(self) -> Dict[str, int]: 
            '''pr_name to pr_idx'''
            if not hasattr(self, '_pr_idx'): 
                print(f'loading index file...')
                with open(f'{FILE_PREFIX}/pr_idx.json') as fp:   # TODO FILE EXISTS? 
                    self._pr_idx = json.load(fp)
            return self._pr_idx
        pr_idx: Dict[str, int] = load_pr_idx(self)
        print(f'count of pr: {len(pr_idx)}')

        text_emb, code_emb = [
            torch.from_numpy(kv_emb_to_tensor(it, pr_idx, device=device))
            for it in [text_emb, code_emb]
        ] 
        # warning: missing embeddings. Total: 379,496; missing: 
        #   * text : 1081
        #   * code : 207,697 (???)

        return torch.cat([text_emb, code_emb], dim=1)

if __name__ == '__main__': 
    klee = PREmbedder()
    # emb = klee.original_code_embedding()

    # def load_pr_idx() -> Dict[str, int]: 
    #     with open(f'{FILE_PREFIX}/pr_idx.json') as fp:
    #         return json.load(fp)
    
    # idx_file = load_pr_idx()

    filepath = f'{FILE_PREFIX}/embedding.pt'
    if os.path.exists(filepath): 
        print('embedding generated. Exit.')
        exit(0)
    emb = klee.merged_embedding(use_old_code_embedding=True)
    torch.save(emb, filepath)