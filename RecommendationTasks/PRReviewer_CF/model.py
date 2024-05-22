#!/usr/bin/env python3 

from config import \
    CF_DICT_FILE_PATH, \
    load_data, load_repo_prs, load_partial_repo_prs

import pickle
import math
import random
import sys, os
from typing import Dict, List, Optional

from tqdm import tqdm



class CollaborativeFiltering: 

    def __init__(self):
        self.repo_sim_matrix: Dict[int, Dict[int, float]] = {}
        self.repo_reviewers: Dict[int, List[int]] = {}

    def generate(self, top_count=0, partial=False): 

        if partial: 
            data = load_partial_repo_prs()  
        else: 
            data = load_repo_prs()

        reviewer_repos: Dict[int, List[int]] = {}
        # for dev_id, repo_id, _ in data:  # dev: developer
        for reviewer_id, repo_id in data:  # dev: developer
            reviewer_repos.setdefault(reviewer_id, [])
            reviewer_repos[reviewer_id].append(repo_id)

        repo_sim_matrix: Dict[int, Dict[int, int]] = {}
        repo_reviewers: Dict[int, List[int]] = {}

        for reviewer, repos in reviewer_repos.items():

            for it in repos: 
                repo_reviewers.setdefault(it, [])
                repo_reviewers[it].append(reviewer)

            size = len(repos)
            for i in range(size):
                for j in range(i + 1, size): 
                    u, v = repos[i], repos[j]

                    repo_sim_matrix.setdefault(u, {})
                    repo_sim_matrix.setdefault(v, {})
                    repo_sim_matrix[u].setdefault(v, 0)
                    repo_sim_matrix[v].setdefault(u, 0)

                    repo_sim_matrix[u][v] += 1
                    repo_sim_matrix[v][u] += 1

        print("Calculating sim between all repos.")
        for repo1 in tqdm(repo_sim_matrix): 
            sims: Dict[int, float] = { 
                repo2: repo_sim_matrix[repo1][repo2] / \
                        math.sqrt(len(repo_reviewers[repo1]) * len(repo_reviewers[repo2]))
                for repo2 in repo_sim_matrix[repo1]
            }
            sims = {k: sims[k] for k in (
                sorted(sims) if top_count == 0 else sorted(sims)[:top_count]
            )}  
            repo_sim_matrix[repo1] = sims


        self.repo_sim_matrix = repo_sim_matrix
        self.repo_reviewers = repo_reviewers

        print('Finish generation')


    def load_pickle(self, filepath: str): 
        with open(filepath, 'rb') as fp: 
            self.repo_sim_matrix, self.repo_reviewers = pickle.load(fp)

    def save_pickle(self, filepath: str): 

        if os.path.exists(filepath): 
            choice = input(f'file "{filepath}" exists. Overwrite it? [Y/n] ')
            if choice not in ['Y', 'y']: 
                return

        with open(filepath, 'wb') as fp:
            pickle.dump((self.repo_sim_matrix, self.repo_reviewers), fp)

    def recommend(self, repo_id: int, search_scope: Optional[List[int]]=None) -> List[int]: 

        recs: List[int]  

        if repo_id not in self.repo_sim_matrix: 
            print("warning: Nothing to recommend.")
            recs = []
        else: 
            related: Dict[int, int] = self.repo_sim_matrix[repo_id]  

            # ret.keys:     repo_id
            # ret.values:   rate of the repo
            ret: Dict[int, float] = {}

            for repo_id2 in related: 
                # use cosine-similarity
                weight = related[repo_id2]
                for reviewer_id in self.repo_reviewers[repo_id2]: 
                    ret.setdefault(reviewer_id, 0)
                    ret[reviewer_id] += weight
            
            ret = sorted(ret.items(), key=lambda it: it[1], reverse=True)
            recs = [it[0] for it in ret]  

        if search_scope is None: 
            return ret  

        others = [it for it in search_scope if it not in recs]
        random.shuffle(others)  
        return recs + others


def train_model(top_count=0, partial=False):

    klee = CollaborativeFiltering()
    klee.generate(top_count, partial)
    postfix = 'full' if top_count == 0 else f'top{top_count}'
    if partial: postfix += '.partial'
    klee.save_pickle(CF_DICT_FILE_PATH + '.' + postfix)
