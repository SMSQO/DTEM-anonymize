import config
import numpy as np

tasks = config.RecommendationTasks
task2model = config.SUFFIX

result = {}
for task in tasks:
    for model in task2model[task]:
        task_model = task + "-" + model
        result[task_model] = []

num_result = 10
num_data = 3        
for result_index in range(num_result):
    result_file = config.RESULT_PREFIX + "{}.txt".format(result_index)
    with open(result_file, 'r') as f:
        for task in tasks:
            for data_index in range(num_data):
                data = f.readline()
                data = [int(x) for x in data.split()]
                i = 0
                for model in task2model[task]:
                    task_model = task + "-" + model
                    result[task_model].extend([data[i], data[i + 1]])
                    i += 2
            f.readline()

for task_model in result:
    data = result[task_model]
    data = sorted(data)
    res = {}
    res["mean"] = np.average(data)
    res["median"] = (data[30] + data[29]) / 2
    res["var"] = np.var(data)
    result[task_model] = res

for it in result:
    print(it, result[it])