#!/usr/bin/env python3 

from .config import \
    TRAIN_DATA_FILE_PATH, CF_DICT_FILE_PATH, \
    load_data, load_repo_contributions

import pickle
import math
import random
import sys, os
from typing import Dict, List, Optional
from tqdm import tqdm


class CollaborativeFiltering: 

    def __init__(self):
        self.user_sim_matrix: Dict[int, Dict[int, float]] = {}
        self.user_repos: Dict[int, List[int]] = {}

    def generate(self, top_count=0): 

        data = load_repo_contributions()  # len(data) = 161241
        # data = load_data(filepath)  # len(data) = 26834

        repo_users: Dict[int, List[int]] = {}
        # for dev_id, repo_id, _ in data:  # dev: developer
        for dev_id, repo_id in data:  # dev: developer
            repo_users.setdefault(repo_id, [])
            repo_users[repo_id].append(dev_id)

        user_sim_matrix: Dict[int, Dict[int, int]] = {}
        user_repos: Dict[int, List[int]] = {}

        for repo, users in repo_users.items():

            for it in users: 
                user_repos.setdefault(it, [])
                user_repos[it].append(repo)

            size = len(users)
            for i in range(size):
                for j in range(i + 1, size): 
                    u, v = users[i], users[j]

                    user_sim_matrix.setdefault(u, {})
                    user_sim_matrix.setdefault(v, {})
                    user_sim_matrix[u].setdefault(v, 0)
                    user_sim_matrix[v].setdefault(u, 0)

                    user_sim_matrix[u][v] += 1
                    user_sim_matrix[v][u] += 1

        for ua in tqdm(user_sim_matrix): 
            sims: Dict[int, float] = { 
                ub: user_sim_matrix[ua][ub] / \
                        math.sqrt(len(user_repos[ua]) * len(user_repos[ub]))
                for ub in user_sim_matrix[ua]
            }
            sims = {k: sims[k] for k in (
                sorted(sims) if top_count == 0 else sorted(sims)[:top_count]
            )}  
            user_sim_matrix[ua] = sims


        self.user_sim_matrix = user_sim_matrix
        self.user_repos = user_repos

        print('Finish generation')


    def load_pickle(self, filepath: str): 
        with open(filepath, 'rb') as fp: 
            self.user_sim_matrix, self.user_repos = pickle.load(fp)

    def save_pickle(self, filepath: str): 

        if os.path.exists(filepath): 
            choice = input(f'file "{filepath}" exists. Overwrite it? [Y/n] ')
            if choice not in ['Y', 'y']: 
                return

        with open(filepath, 'wb') as fp:
            pickle.dump((self.user_sim_matrix, self.user_repos), fp)

    def recommend(self, dev_id: int, search_scope: Optional[List[int]]=None) -> List[int]: 

        recs: List[int]   

        if dev_id not in self.user_sim_matrix: 
            print("warning: Nothing to recommend.")
            recs = []
        else: 
            related: Dict[int, int] = self.user_sim_matrix[dev_id]  

            ret: Dict[int, float] = {}

            for dev_id2 in related: 
                weight = related[dev_id2]
                for repo_id in self.user_repos[dev_id2]: 
                    ret.setdefault(repo_id, 0)
                    ret[repo_id] += weight
            
            ret = sorted(ret.items(), key=lambda it: it[1], reverse=True)
            recs = [it[0] for it in ret]  

        if search_scope is None: 
            return ret  

        others = [it for it in search_scope if it not in recs]
        random.shuffle(others) 
        return recs + others


def train_model(top_count=0):

    klee = CollaborativeFiltering()
    klee.generate(top_count)
    postfix = 'full' if top_count == 0 else f'top{top_count}'
    klee.save_pickle(CF_DICT_FILE_PATH + '.' + postfix)
