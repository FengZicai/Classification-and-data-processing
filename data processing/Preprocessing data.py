import json
import cv2


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


base = 'E:/AI challenger/train_simplify/'
for i in range(61):
    mkdir(base + str(i))


# path = 'E:/AI challenger/Alexnet/AgriculturalDisease_validationset/images/'
#
# with open('AgriculturalDisease_validation_annotations.json', 'r') as f:
#     train_list = json.load(f)
# for line in train_list:
#     imageOriginal = cv2.imread(path + line['image_id'])
#     cv2.imwrite(base + str(line['disease_class']) + '/' + line['image_id'], imageOriginal)
