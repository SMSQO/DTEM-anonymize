import config
import json
import random

num_selection = 30

data = {
    'ContributionRepo' : [],
    'PRReviewer': [],
    'RepoMaintainer': [],
    'SimDeceloper': []
}

contributors = {}
repositories = {}
prs = {}

with open(config.CONTRIBUTOR, 'r') as f:
    contributors = json.load(f)
    contributors = {int(contributors[key]) : key for key in contributors}
with open(config.REPOSITORY, 'r') as f:
    repositories = json.load(f)
    repositories = {int(repositories[key]) : key for key in repositories}
with open(config.PRS, 'r') as f:
    prs = json.load(f)
    prs = {int(prs[key]) : key for key in prs}
    

for task in config.RecommendationTasks:
    dstFile = config.SELECTED_DATA[task]
    result = []
    
    for model in config.SUFFIX[task]:
        srcFile = config.PREFIX[task] + model + '.json'
        data = {}
        with open(srcFile, 'r') as f:
            data = json.load(f)
        
        selected_data = []
        for key_id in data:
            recommended_id = data[key_id][1]
            
            if len(recommended_id) == 0:
                continue
            selected_data.append([int(key_id), recommended_id[0], model])
            
        selected_data = random.choices(selected_data, k=num_selection)
        result.extend(selected_data)
            
    if task == 'ContributionRepo':
        result = [[id, contributors[id], rec_id, repositories[rec_id], model] for id, rec_id, model in result]
    elif task == 'PRReviewer':
        result = [[id, prs[id], rec_id, contributors[rec_id], model] for id, rec_id, model in result]
    elif task == 'RepoMaintainer':
        result = [[id, repositories[id], rec_id, contributors[rec_id], model] for id, rec_id, model in result]
    elif task == 'SimDeveloper':
        result = [[id, contributors[id], rec_id, contributors[rec_id], model] for id, rec_id, model in result]

    with open(dstFile, 'w') as f:
        json.dump(result, f, indent=4)