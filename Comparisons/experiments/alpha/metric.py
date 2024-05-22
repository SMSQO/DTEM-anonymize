import os
import json
import numpy as np

'''
    BAD MANNER! 
        RecommendationTasks/SimDeveloper/metric/metric.py 
'''

from RecommendationTasks.SimDeveloper.metric.metric import analysis
from ..general import load_yaml_cfg


cfg = load_yaml_cfg()['alpha']
task_cfg = cfg['tasks']['sim_developer']

def main(): 

    src_file = task_cfg['result']['valid_test_result']
    with open(src_file, "r", encoding="utf-8") as inf:
        src = json.load(inf)
    
    total = len(src)
    scope_length = 0

    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    MRR = 0

    for item in src:
        topks = src[item][0]
        tmp1, tmp3, tmp5, tmp10, mrr = analysis(topks)
        top1 += tmp1
        top3 += tmp3
        top5 += tmp5
        top10 += tmp10
        MRR += mrr
        scope_length += len(src[item][1])

    print(top1 / total, top3 / total, top5 / total, top10 / total, MRR / total)
    print(scope_length / total)

main()