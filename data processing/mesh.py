import os
import json

path = "/home/new/ResNet/test22656/"
filelist = os.listdir(path)
result = []
submit = './submit_test22656.json'
for file in filelist:
    with open(path + file, 'r') as f:
        train_list = json.load(f)
    for i in range(len(train_list)):
        result.append(train_list[i])
with open(submit, 'w') as f:
    json.dump(result, f)
