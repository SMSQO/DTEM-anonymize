from bdb import effective
import os
import json
import sys

validation_dataset = "./dataset_valid_test_modified.json"
with open(validation_dataset, "r", encoding="utf-8") as inf:
        dataset = json.load(inf)

GT_info = {}
search_scope_info = {}
contributor_info = {}

lengths_of_GT = []
lengths_of_search_scope = []

for contributor_idx, search_scope, labels in dataset:
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
contributor_info['length'] = len(dataset)

print(GT_info)                  # {'avg_length': 230.91674573055028}
print(search_scope_info)        # {'avg_length': 45.2685009487666}
print(contributor_info)         # {'length': 4953}

# According to the paper,
# filter out contributors with less than 10 candidatess.
lengths_of_GT = []
lengths_of_search_scope = []
length_of_contributors = 0

for contributor_idx, search_scope, labels in dataset:
    if(len(labels) < 5):
        continue
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))
    length_of_contributors += 1

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
contributor_info['length'] = length_of_contributors

print(GT_info)                  # {'avg_length': 278.0}
print(search_scope_info)        # {'avg_length': 49.706926159129935}
print(contributor_info)         # {'length': 4216}

# On average, recommend repos from 278 candidates to hit 49 GTs.


# look at how many gts are actually in search_scope，
fake_gt = 0
length_of_contributors = 0
lengths_of_GT = []

for contributor_idx, search_scope, labels in dataset:
    if len(labels) < 5:
        continue
    length_of_contributors += 1
    effective_gt = list(set(search_scope).intersection(set(labels)))
    if len(effective_gt) == 0:
        fake_gt += 1
    lengths_of_GT.append(len(effective_gt))

print(length_of_contributors)
print(fake_gt)
print(sum(lengths_of_GT) / len(lengths_of_GT))
    
    