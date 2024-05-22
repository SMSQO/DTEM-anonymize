#!/usr/bin/env python3 


import numpy as np
from typing import *
import torch
from tqdm import tqdm
import os

class AggregateDeveloperEmbedding: 

    def __init__(self, contributor_count: int): 
        self.contributor_count = contributor_count

    def merged_embedding(self, filepath: Optional[str]) -> torch.FloatTensor: 
 
        if filepath is not None and os.path.exists(filepath): 
            print(f'Contributor embedding already exists, loading.')
            return torch.load(filepath)
        print(f'Contributor embedding not found. Start loading and source and merging.')

        pr_emb, issue_emb, repo_emb = \
                self.load_contributor_pr_embedding(), \
                self.load_contributor_issue_embedding(), \
                self.load_contributor_repo_embedding()
        
        ret = torch.cat([pr_emb, issue_emb, repo_emb], dim=1)

        # region testing part
        zeros = 0 
        for it in ret: 
            if (it ** 2).sum().item() == 0: 
                zeros += 1
        print(f'{zeros = }')
        # endregion

        torch.save(ret, filepath)
        return ret  # zeros = 1211. 
        # ret.shape = (394474, 14336)
        # Really big tensor, loading time: 8.156s 


    def load_contributor_pr_embedding(self) -> torch.FloatTensor: 
        return self._load_embedding(
            emb_filepath='data/pr_embedding/embedding.pt',
            rel_filename='contributor_propose_pr.txt',
            handler=lambda it: (int(it[0]), int(it[1]), 1)
        ) # zeros: 298,942 (394,474, 75.78%)

    def load_contributor_issue_embedding(self) -> torch.FloatTensor: 
        return self._load_embedding(
            emb_filepath='data/issue_embedding/embedding.pt',
            rel_filename='contributor_propose_issue.txt',
            handler=lambda it: (int(it[0]), int(it[1]), 1)
        ) # zeros: 107,661 (27.29%)

    def load_contributor_repo_embedding(self) -> torch.FloatTensor: 
        # There are three kinds of relationships between contributors and repositories: 
        # star, watch, commit(contribute) 
        # we give star and watch a weight 1, and commit a weight of 1 / COMMIT_AVG.
        COMMIT_AVG = 180.7840623662716  # average commit count for each contributor commit to a repo
        FILE_PREFIX = 'data/relationships'

        rel_star, handler_star  = \
                self._load_relationship(f'{FILE_PREFIX}/contributor_star_repo.txt'), \
                lambda it: (int(it[0]), int(it[1]), 1)
        rel_watch, handler_watch = \
                self._load_relationship(f'{FILE_PREFIX}/contributor_watch_repo.txt'), \
                lambda it: (int(it[0]), int(it[1]), 1)
        rel_contribute, handler_contribute = \
                self._load_relationship(f'{FILE_PREFIX}/contributor_commit_repo.txt'), \
                lambda it: (int(it[0]), int(it[1]), float(it[2]) / COMMIT_AVG)

        # @bad manner: duplicate from self._load_embedding
        repo_emb: torch.FloatTensor = torch.load('data/repo_embedding/embedding.pt')
        size, emb_size = self.contributor_count, repo_emb.shape[1]
        ret, cnt = torch.zeros((size, emb_size)), np.zeros((size,)) 
        print(f'{ret.shape = }')

        for relationship, handler in tqdm([
            (rel_star, handler_star), 
            (rel_watch, handler_watch), 
            (rel_contribute, handler_contribute)
        ]): 
            for it in relationship: 
                contributor_id, elem_id, weight = handler(it)
                ret[contributor_id] += repo_emb[elem_id]
                cnt[contributor_id] += weight
        
        zeros = 0
        for i in range(ret.shape[0]):
            if cnt[i] == 0:  
                zeros += 1
                continue
            ret[i] /= cnt[i]
        print(f'{zeros = }')
        return ret # zeros: 181,627 (46.04%)


    def _load_embedding(
        self,
        *,
        emb_filepath: str,
        rel_filename: str,
        handler: Callable[[List[str]], Tuple[int, int, int]]
    ) -> torch.FloatTensor: 

        '''
        @see also: 
            GNN/DataPreprocess/2.build_structure_graph.py GraphBuilder
        '''
        filepath = f'data/relationships/{rel_filename}'
        print(f'loading {filepath = }')
        relationship = self._load_relationship(filepath)
        elem_emb: torch.FloatTensor = torch.load(emb_filepath)
        print(elem_emb)

        size, emb_size = self.contributor_count, elem_emb.shape[1] 
        ret, cnt = torch.zeros((size, emb_size)), np.zeros((size,)) 
        print(f'{ret.shape = }')

        for it in relationship: 
            contributor_id, elem_id, weight = handler(it)
            ret[contributor_id] += elem_emb[elem_id]
            cnt[contributor_id] += weight

        zeros = 0
        for i in range(ret.shape[0]):
            if cnt[i] == 0:  
                zeros += 1
                continue
            ret[i] /= cnt[i]
        print(f'{zeros = }')
        return ret
        

    def _load_relationship(self, filepath: str) -> List[List[str]]: 
 
        def seq() -> Iterable[List[str]]:
            with open(filepath) as fp:  
                for line in fp: 
                    yield [it for it in line.split('\t')]
        return list(seq())
    

if __name__ == '__main__': 
    klee = AggregateDeveloperEmbedding(contributor_count=394_474)
    ret = klee.merged_embedding('data/contributor_embedding/embedding.pt')