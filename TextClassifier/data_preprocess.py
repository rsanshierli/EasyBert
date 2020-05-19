import os
from tqdm import tqdm
import numpy as np

def convert(content):
    content = content.replace("\n", "")
    content = content.replace("\u3000", "")
    content = content.replace(" ", "")
    content = content.replace("\xa0", "")
    content = content.replace("\t", "")

    str2list = list(content)
    if len(str2list) <= 256:
        return content
    else:
        list2str = "".join(content[:256])
        return list2str

def get_data(file_path, model):
    paths = os.listdir(file_path)
    tmp = open(f'{model}.txt', 'w', encoding='utf-8')
    for t in tqdm(range(len(paths))):
        p = paths[t]
        with open(f'{file_path}/{p}', encoding='utf-8') as f:
            tmp.write(convert(f.read()))
            tmp.write('\n')
    tmp.close()

get_data('./THUCNews/灾难', '灾难')
'''
    没写遍历。。。
'''



import os
from tqdm import tqdm
def get_all_data(file_path, model):
    class_dict = {}
    paths = os.listdir(file_path)
    tmp = open(f'{model}.txt', 'w', encoding='utf-8')
    for t in tqdm(range(len(paths))):
        p = paths[t]
        class_dict[p] = t
        with open(f'{file_path}/{p}', encoding='utf-8') as f:
            text = f.readlines()
            for i in text:
                if len(i) != 0:
                    tmp.write("".join(i.split('\n')[0]) + '\t' + str(class_dict[p]))
                    tmp.write('\n')
    tmp.close()
    with open('classes.txt', 'w', encoding='utf-8') as f:
        for i in class_dict.items():
            print('{}======>{}'.format(list(i)[0].split('.')[0], list(i)[1]))
            f.write(str(list(i)[0].split('.')[0]))
            f.write('\n')

'''
    体育======>0
    军事======>1
    娱乐======>2
    政治======>3
    教育======>4
    灾难======>5
    社会======>6
    科技======>7
    财经======>8
    违法======>9
'''
file_path = 'D:/任鹏程/Text-Classification/THUCNews/处理数据/'
get_all_data(file_path, 'all_data')


def random_split(data):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0 and i % 10 != 1]
    valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
    test_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 1]
    return train_data, valid_data, test_data


with open('all_data.txt',  'r', encoding='utf-8') as f:
    text = f.readlines()

train_data, valid_data, test_data = random_split(text)


with open("train.txt", 'w', encoding='utf-8') as f:
    for line in train_data:
        content = line.split('\t')[0]
        label = line.split('\t')[1]
        f.write(content + '\t' + label)
with open("dev.txt", 'w', encoding='utf-8') as f:
    for line in valid_data:
        content = line.split('\t')[0]
        label = line.split('\t')[1]
        f.write(content + '\t' + label)
with open("test.txt", 'w', encoding='utf-8') as f:
    for line in test_data:
        content = line.split('\t')[0]
        label = line.split('\t')[1]
        f.write(content + '\t' + label)