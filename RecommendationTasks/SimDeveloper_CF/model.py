#!/usr/bin/env python3 

from config import \
    CF_DICT_FILE_PATH, \
    load_data, load_partial_user1_user2

import pickle
import math
import random
import sys, os
from typing import Dict, List, Optional

from tqdm import tqdm


class CollaborativeFiltering: 

    def __init__(self):
        self.user1_sim_matrix: Dict[int, Dict[int, float]] = {}
        self.user1_user2s: Dict[int, List[int]] = {}

    def generate(self, top_count=0, partial=False): 

        if partial: 
            data = load_partial_user1_user2()  

        user2_user1s: Dict[int, List[int]] = {}

        for user1, user2 in data:
            user2_user1s.setdefault(user2, [])
            user2_user1s[user2].append(user1)
            user2_user1s.setdefault(user1, [])
            user2_user1s[user1].append(user2)

        user1_sim_matrix: Dict[int, Dict[int, int]] = {}
        user1_user2s: Dict[int, List[int]] = {}

        for user2, user1s in user2_user1s.items():

            for it in user1s: 
                user1_user2s.setdefault(it, [])
                user1_user2s[it].append(user2)

            size = len(user1s)
            for i in range(size):
                for j in range(i + 1, size): 
                    u, v = user1s[i], user1s[j]

                    user1_sim_matrix.setdefault(u, {})
                    user1_sim_matrix.setdefault(v, {})
                    user1_sim_matrix[u].setdefault(v, 0)
                    user1_sim_matrix[v].setdefault(u, 0)

                    user1_sim_matrix[u][v] += 1
                    user1_sim_matrix[v][u] += 1


        print("Calculating sim between all repos.")
        for user1 in tqdm(user1_sim_matrix): 
            sims: Dict[int, float] = { 
                user2: user1_sim_matrix[user1][user2] / \
                        math.sqrt(len(user1_user2s[user1]) * len(user1_user2s[user2]))
                for user2 in user1_sim_matrix[user1]
            }
            sims = {k: sims[k] for k in (
                sorted(sims) if top_count == 0 else sorted(sims)[:top_count]
            )}  
            user1_sim_matrix[user1] = sims


        self.user1_sim_matrix = user1_sim_matrix
        self.user1_user2s = user1_user2s

        print('Finish generation')


    def load_pickle(self, filepath: str): 
        with open(filepath, 'rb') as fp: 
            self.user1_sim_matrix, self.user1_user2s = pickle.load(fp)

    def save_pickle(self, filepath: str): 

        with open(filepath, 'wb') as fp:
            pickle.dump((self.user1_sim_matrix, self.user1_user2s), fp)

    def recommend(self, user1_id: int, search_scope: Optional[List[int]]=None) -> List[int]: 

        recs: List[int]  

        if user1_id not in self.user1_sim_matrix: 
            # print("warning: Nothing to recommend.")
            recs = []
        else: 
            
            related: Dict[int, int] = self.user1_sim_matrix[user1_id]  

            ret: Dict[int, float] = {}

            for user1_id2 in related: 
                # use cosine-similarity
                weight = related[user1_id2]
                for user2_id in self.user1_user2s[user1_id2]: 
                    ret.setdefault(user2_id, 0)
                    ret[user2_id] += weight
            
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
