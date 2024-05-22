import os
import json
import sys

validation_dataset = "./dataset_valid_test_modified.json"
with open(validation_dataset, "r", encoding="utf-8") as inf:
        dataset = json.load(inf)

GT_info = {}
search_scope_info = {}
pr_info = {}

lengths_of_GT = []
lengths_of_search_scope = []

for repo_idx, pr_idx, search_scope, labels in dataset:
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
pr_info['length'] = len(dataset)

print(GT_info)                  # {'avg_length': 1.665868607546447}
print(search_scope_info)        # {'avg_length': 17.706665389772073}
print(pr_info)                  # {'length': 10442}

# According to the paper,
# filter out contributors with less than 10 candidatess.
lengths_of_GT = []
lengths_of_search_scope = []
length_of_prs = 0

for repo_idx, pr_idx, search_scope, labels in dataset:
    if(len(search_scope) < 10):
        continue
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))
    length_of_prs += 1

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
pr_info['length'] = length_of_prs

print(GT_info)                  # {'avg_length': 1.861699979810216}
print(search_scope_info)        # {'avg_length': 32.45245305875227}
print(pr_info)                  # {'length': 4953}

# On average, recommend repos from 32 candidates to hit 1.8 GTs.


# look at if there is any gt not in search_scope, i.e. reviewer himself did not commit

gt_not_in_scope = 0
cnt = 0
fake_gt = 0

for repo_idx, pr_idx, search_scope, labels in dataset:
    if len(search_scope) < 10: continue     # only look at these instances
    cnt += 1
    for label in labels:
        if label not in search_scope:
            gt_not_in_scope += 1
    if len(list(set(search_scope).intersection(set(labels)))) == 0:
        fake_gt += 1
            
        
print(gt_not_in_scope)                      # 1961
print(gt_not_in_scope / cnt)                # 0.3959
print(fake_gt)                              # 631
print(len(dataset))                         # 10442