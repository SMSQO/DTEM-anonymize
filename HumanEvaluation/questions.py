import json
import config

selected_data = {}
num_selection = 30

for task in config.RecommendationTasks: 
    file = config.SELECTED_DATA[task]
    with open(file, 'r') as f:
        data = json.load(f)
        data = {task : data}
        selected_data.update(data)
        
for task in config.RecommendationTasks:
    model_data = {}
    for item in selected_data[task]:
        if item[4] not in model_data:
            model_data[item[4]] = []
        model_data[item[4]].append([item[i] for i in range(4)])
    selected_data[task] = model_data


for i in range(num_selection):
    questionaire = {}
    questionaire['id'] = i
    questionaire['questions'] = []
    for task in config.RecommendationTasks:
        for model in selected_data[task]:
            question = selected_data[task][model][i]
            question.append(task)
            question.append(model)
            questionaire['questions'].append(question)
    
    with open('./questionnaire/data/{}.json'.format(i), 'w') as f:
        json.dump(questionaire, f, indent=4)
       