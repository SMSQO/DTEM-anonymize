#!/usr/bin/env python3

from RecommendationTasks.ContributionRepo_CF.config import \
        CF_DICT_FILE_PATH, VALID_TEST_DATA_FILE_PATH, VALID_RESULT_PATH, \
        load_data

from RecommendationTasks.ContributionRepo_CF.model  \
    import CollaborativeFiltering

import json
from typing import List, Dict, Tuple


def evaluate(model_postfix='full'): 

    klee = CollaborativeFiltering()
    klee.load_pickle(CF_DICT_FILE_PATH + '.' + model_postfix)

    dataset = load_data(VALID_TEST_DATA_FILE_PATH)

    result: Dict[int, Tuple[List[int], List[int]]] = {}

    for dev_id, search_scope, gt in dataset: 
        if len(gt) < 5: continue

        # Exists dev_id, s.t. len(recs) < 20. Be careful. 
        # (sometimes happens when len(search_scope) < 20)
        # recs: recommendations
        recs: List[int] = klee.recommend(dev_id, search_scope)[:20]

        acc: List[int] = [0] * 21
        for i in range(20):
            acc[i + 1] = acc[i] + (i < len(recs) and recs[i] in gt)

        result[dev_id] = (acc, search_scope)

    with open(VALID_RESULT_PATH + '.' + model_postfix, 'w') as fp:
        json.dump(result, fp)