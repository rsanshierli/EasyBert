import re
import pandas as pd
from sklearn.utils import shuffle

def clean(text):
    # text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    # text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)  # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    text = text.replace("转发微博", "")  # 去除无意义的词语
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    return text.strip()

def datasets(dataframe, test=False):
    dataframe = dataframe.drop_duplicates(subset='微博中文内容')
    dataframe.reset_index(drop=True, inplace=True)
    text = []
    if test == False:
        dataframe = dataframe.dropna(subset=['情感倾向'])
        dataframe.reset_index(drop=True, inplace=True)
        for i in range(len(dataframe)):
            content = clean(str(dataframe.loc[i, '微博中文内容']))
            label = dataframe.loc[i, '情感倾向']
            if label == -1:
                label = 2
            text.append((content, int(label)))

        text = shuffle(text, random_state=1)
        test_proportion = 0.05
        test_idx = int(len(text) * test_proportion)

        test_data = text[:test_idx]
        train_data = text[test_idx:]

        return train_data, test_data
    elif test == True:
        for i in range(len(dataframe)):
            text.append(clean(str(dataframe.loc[i, "微博中文内容"])))
        return text

if __name__ == '__main__':
    train_path = ''
    test_path = ''
    data = pd.read_csv(train_path, encoding='utf-8')
    test = pd.read_csv(test_path, encoding='utf-8')
    train_data, test_data = datasets(data, test=False)
    predict_data = datasets(test, test=True)
    with open("./Sentiment/data/train.txt", "a", encoding="utf-8") as f:
        for line in train_data:
            f.write(str(line[0]) + '\t' + str(line[1]))
            f.write("\n")
    with open("./Sentiment/data/dev.txt", "a", encoding="utf-8") as f:
        for line in test_data:
            f.write(str(line[0]) + '\t' + str(line[1]))
            f.write("\n")
    with open("./Sentiment/data/test.txt", "a", encoding='utf-8') as f:
        for line in predict_data:
            f.write(str(line))
            f.write("\n")

    print('Finish!')