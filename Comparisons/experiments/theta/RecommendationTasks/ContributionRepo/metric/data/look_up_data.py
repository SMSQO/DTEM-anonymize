import os
import json
import sys

validation_dataset = "./dataset_valid_test.json"
with open(validation_dataset, "r", encoding="utf-8") as inf:
        dataset = json.load(inf)

GT_info = {}
search_scope_info = {}
contributor_info = {}

lengths_of_GT = []
lengths_of_search_scope = []

for contrbutor_idx, search_scope, labels in dataset:
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
contributor_info['length'] = len(dataset)

print(GT_info)                  # {'avg_length': 1.22041181736795}
print(search_scope_info)        # {'avg_length': 47.87520143240824}
print(contributor_info)         # {'length': 5585}

# According to the paper,
# filter out contributors with less than 5 GTs.
lengths_of_GT = []
lengths_of_search_scope = []
length_of_contributor = 0

for contrbutor_idx, search_scope, labels in dataset:
    if(len(labels) < 5):
        continue
    lengths_of_GT.append(len(labels))
    lengths_of_search_scope.append(len(search_scope))
    length_of_contributor += 1

GT_info['avg_length'] = sum(lengths_of_GT) / len(lengths_of_GT)
search_scope_info['avg_length'] = sum(lengths_of_search_scope) / len(lengths_of_search_scope)
contributor_info['length'] = length_of_contributor

print(lengths_of_GT)            # [5, 8, 5, 5, 8, 15, 5, 6, 11, 5, 5, 7, 5, 10, 6, 5, 5, 6, 9, 5, 6, 5, 7, 7, 5, 5, 5, 5, 22, 8, 5, 8, 56, 5, 8, 10, 5, 5, 7, 5, 26, 5]
print(GT_info)                  # {'avg_length': 8.476190476190476}
print(lengths_of_search_scope)  # [121, 213, 829, 35, 208, 555, 53, 70, 293, 461, 107, 342, 148, 412, 107, 22, 35, 461, 272, 46, 24, 93, 371, 338, 221, 10, 11, 390, 750, 105, 114, 138, 474, 114, 226, 83, 356, 90, 132, 210, 67, 10]
print(search_scope_info)        # {'avg_length': 217.07142857142858}
print(contributor_info)         # {'length': 42}

# On average, recommend repos from 217 candidates to hit 8 GTs.
