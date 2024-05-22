import os
import json
import sys

validation_dataset = "./dataset_valid_test.json"
with open(validation_dataset, "r", encoding="utf-8") as inf:
        dataset = json.load(inf)

GT_info = {}
search_scope_info = {}
repo_info = {}

lengths_of_GT = []
lengths_of_search_scope = []

for repo_idx, search_scope, labels in dataset:
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
repo_info['length'] = len(dataset)

print(GT_info)                  # {'avg_length': 1.0}
print(search_scope_info)        # {'avg_length': 1.388618874038072}
print(repo_info)                  # {'length': 4938}

# According to the paper,
# filter out contributors with less than 10 candidatess.
lengths_of_GT = []
lengths_of_search_scope = []
length_of_repos = 0

for repo_idx, search_scope, labels in dataset:
    if(len(search_scope) < 10):
        continue
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))
    length_of_repos += 1

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
repo_info['length'] = length_of_repos

print(GT_info)                  # {'avg_length': 1.0}
print(search_scope_info)        # {'avg_length': 17.771428571428572}
print(repo_info)                  # {'length': 35}

# On average, recommend repos from 1.8 candidates to hit 32 GTs.


# look at if there is any gt not in search_scope, i.e. owner himself did not commit

gt_not_in_scope = 0
gt_not_in_scope_with_enough_scope = 0

for repo_idx, search_scope, labels in dataset:
    if len(labels) > 1:
        print("more than 1 owner")
    if labels[0] not in search_scope:
        gt_not_in_scope += 1
        if len(search_scope) >= 10:
            gt_not_in_scope_with_enough_scope += 1
print(gt_not_in_scope)                      # 516
print(gt_not_in_scope_with_enough_scope)    # 0
print(len(dataset))                         # 4938