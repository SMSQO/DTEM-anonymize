#!/usr/bin/env python

import os, json
from typing import Dict, Tuple, List

from tqdm import tqdm


FILE_PREFIX = 'RecommendationTasks/SimDeveloper_CF'

# MAYBE DEPRECATED
CF_DICT_FILE_PATH = 'RecommendationTasks/SimDeveloper_CF/bin/cf_dict.pkl'

VALID_TEST_DATA_FILE_PATH   = f'{FILE_PREFIX}/metric/data/dataset_valid_test_modified.json'
VALID_RESULT_PATH           = f'{FILE_PREFIX}/metric/result/result_valid_test_modified.json'

REPO_PARTIAL_FILE_PATH      = 'RecommendationTasks/SimDeveloper_CF/data/train.json'
REPOSITORY_FILE_PATH        = 'GNN/DataPreprocess/full_graph/content/repositories.json'
MAINTAINER_FILE_PATH        = 'GNN/DataPreprocess/full_graph/content/contributors.json'


# check files exists
for it in {
    VALID_TEST_DATA_FILE_PATH,
    REPOSITORY_FILE_PATH,
    MAINTAINER_FILE_PATH,
}: 
    assert os.path.exists(it)


def load_data(filepath: str) -> List[Tuple[int, int, int]]: 
    with open(filepath) as fp: 
        return json.load(fp)

def load_partial_user1_user2() -> List[Tuple[int, int]]:
    with open(REPO_PARTIAL_FILE_PATH) as fp:
        prs = json.load(fp)
    ret: List[Tuple[int, int]] = []
    print('Loading partial PRs')
    for src_user, pos_user, neg_user in prs:
        ret.append([src_user, pos_user])
    return ret

