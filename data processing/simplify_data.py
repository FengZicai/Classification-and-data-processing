# author by LYS 2017/5/24
# for Deep Learning course
'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil


def copyFile(fileDir,tarDir):
    # 1
    pathDir = os.listdir(fileDir)

    # 2
    sample = random.sample(pathDir, 300)
    print(sample)

    # 3
    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)


if __name__ == '__main__':
    fileDir = "E:\\AI challenger\\train_change_simplify\\"
    tarDir = "E:\\AI challenger\\train_simplify\\"
    for i in range(61):
        filepath = fileDir + str(i) + '\\'
        tarpath = tarDir + str(i) + '\\'
        pathDir = os.listdir(filepath)
        sample = random.sample(pathDir, 300)
        for name in sample:
            shutil.copyfile(filepath + name, tarpath + name)
