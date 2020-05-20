import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from TextMatch.Bert.model import BertModelTest
from TextMatch.Bert.utils import predict
from TextMatch.Bert.data import DataPrecessForSentence

def main(test_file, batch_size=1):
    pretrained_file = './TextMatch/Bert/models/best.pth.tar'
    pretrained_model = './TextMatch/pretrained_model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    test_data = DataPrecessForSentence(bert_tokenizer, test_file, pred=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    model = BertModelTest(pretrained_model).to(device)
    model.load_state_dict(checkpoint['model'])
    result = predict(model, test_file, test_loader, device)

    return result

if __name__ == '__main__':
    '''
    原数据集标签：0, 1, 0, 1, 0
    预测结果：
    ========== Predict Result ==========
    ['谁有狂三这张高清的', '这张高清图，谁有', '相似']
    ['英雄联盟什么英雄最好', '英雄联盟最好英雄是什么', '不相似']
    ['这是什么意思，被蹭网吗', '我也是醉了，这是什么意思', '不相似']
    ['现在有什么动画片好看呢？', '现在有什么好看的动画片吗？', '不相似']
    ['请问晶达电子厂现在的工资待遇怎么样要求有哪些', '三星电子厂工资待遇怎么样啊', '相似']
    '''

    text = [['谁有狂三这张高清的', '这张高清图，谁有'],
            ['英雄联盟什么英雄最好', '英雄联盟最好英雄是什么'],
            ['这是什么意思，被蹭网吗', '我也是醉了，这是什么意思'],
            ['现在有什么动画片好看呢？', '现在有什么好看的动画片吗？'],
            ['请问晶达电子厂现在的工资待遇怎么样要求有哪些', '三星电子厂工资待遇怎么样啊']]
    result = main(text)
    print(10*"=", "Predict Result", 10*"=")
    print(result)