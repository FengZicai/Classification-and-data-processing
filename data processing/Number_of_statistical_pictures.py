import os
import csv

path = 'E:/AI challenger/train_change_simplify'
path_list = os.listdir(path)
#print(path_list)
path_list.sort(key=lambda x: int(x[:]))

datas = []
for folder in path_list:
    number = []
    number.append(folder)
    filelist = os.listdir(path + '/' + folder)
    number.append(len(filelist))
    datas.append(number)
with open('train_change_simplify.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in datas:
        writer.writerow(row)





# #读取csv
# f = open('test.csv',mode='r',encoding='gbk')   #mode读取模式，采用b的方式处理可以省去很多问题，encoding编码方式
# reader = csv.reader(f)  #获取输入数据。把每一行数据转化成了一个list，list中每个元素是一个字符串
# for row in reader:  #按行读取文件。一行读取为字符串，在使用分割符（默认逗号）分割成字符串列表，对于包含逗号，并使用""标志的字符串不进行分割
#      print(row)
#      print(type(row))
# f.close()
