import os
import json

path = "/home/smile/ResNet/train_change"
filelist = os.listdir(path)
disease_class = []
result = []
submit = './train.json'
for file in filelist:
    
    img_path = os.path.join(path, file)
    img_list = os.listdir(img_path)
    for i in range(len(img_list)):
        temp_dict = {}
        temp_dict['image_id'] = img_list[i]
        temp_dict['disease_class'] = file
        result.append(temp_dict)
with open(submit, 'w') as f:
    json.dump(result, f)
