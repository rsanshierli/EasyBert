import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertModelTest
from utils import predict
from data import DataPrecessForSentence

def main(test_file, pretrained_file, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_tokenizer = BertTokenizer.from_pretrained('/models/vocabs.txt', do_lower_case=True)
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    test_data = DataPrecessForSentence(bert_tokenizer, test_file, pred=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint['model'])
    result = predict(model, test_file, test_loader)

    return result

if __name__ == '__main__':
    text = []
    result = main(text, 'models/best.pth.tar')
    print(10*"=", "Predict Result", 10*"=")
    print(result)