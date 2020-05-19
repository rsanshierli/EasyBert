import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'bert'
        self.class_list = ['体育', '军事', '娱乐', '政治', '教育', '灾难', '社会', '科技', '财经', '违法']          # 类别名单
        self.save_path = './THUCNews/saved_dict/bert.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

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

def load_dataset(data, config):
    pad_size = config.pad_size
    contents = []
    for line in data:
        lin = convert(line)
        token = config.tokenizer.tokenize(lin)      # 分词
        token = [CLS] + token                           # 句首加入CLS
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, int(0), seq_len, mask))
    return contents

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches     # data
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):     # 返回下一个迭代器对象，必须控制结束条件
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):     # 返回一个特殊的迭代器对象，这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成。
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, 1, config.device)
    return iter

def match_label(pred, config):
    label_list = config.class_list
    return label_list[pred]


def final_predict(config, model, data_iter):
    map_location = lambda storage, loc: storage
    model.load_state_dict(torch.load(config.save_path, map_location=map_location))
    model.eval()
    predict_all = np.array([])
    with torch.no_grad():
        for texts, _ in data_iter:
            outputs = model(texts)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            pred_label = [match_label(i, config) for i in pred]
            predict_all = np.append(predict_all, pred_label)

    return predict_all

def main(text):
    config = Config()
    model = Model(config).to(config.device)
    test_data = load_dataset(text, config)
    test_iter = build_iterator(test_data, config)
    result = final_predict(config, model, test_iter)
    for i, j in enumerate(result):
        print('text:{}'.format(text[i]))
        print('label:{}'.format(j))

if __name__ == '__main__':

    test = ['国考28日网上查报名序号查询后务必牢记报名参加2011年国家公务员的考生，如果您已通过资格审查，那么请于10月28日8：00后，登录考录专题网站查询自己的“关键数字”——报名序号。'
            '国家公务员局等部门提醒：报名序号是报考人员报名确认和下载打印准考证等事项的重要依据和关键字，请务必牢记。此外，由于年龄在35周岁以上、40周岁以下的应届毕业硕士研究生和'
            '博士研究生(非在职)，不通过网络进行报名，所以，这类人报名须直接与要报考的招录机关联系，通过电话传真或发送电子邮件等方式报名。',
            '高品质低价格东芝L315双核本3999元作者：徐彬【北京行情】2月20日东芝SatelliteL300(参数图片文章评论)采用14.1英寸WXGA宽屏幕设计，配备了IntelPentiumDual-CoreT2390'
            '双核处理器(1.86GHz主频/1MB二级缓存/533MHz前端总线)、IntelGM965芯片组、1GBDDR2内存、120GB硬盘、DVD刻录光驱和IntelGMAX3100集成显卡。目前，它的经销商报价为3999元。',
            '国安少帅曾两度出山救危局他已托起京师一代才俊新浪体育讯随着联赛中的连续不胜，卫冕冠军北京国安的队员心里到了崩溃的边缘，俱乐部董事会连夜开会做出了更换主教练洪元硕的决定。'
            '而接替洪元硕的，正是上赛季在李章洙下课风波中同样下课的国安俱乐部副总魏克兴。生于1963年的魏克兴球员时代并没有特别辉煌的履历，但也绝对称得上特别：15岁在北京青年队获青年'
            '联赛最佳射手，22岁进入国家队，著名的5-19一战中，他是国家队的替补队员。',
            '汤盈盈撞人心情未平复眼泛泪光拒谈悔意(附图)新浪娱乐讯汤盈盈日前醉驾撞车伤人被捕，原本要彩排《欢乐满东华2008》的她因而缺席，直至昨日(12月2日)，盈盈继续要与王君馨、马'
            '赛、胡定欣等彩排，大批记者在电视城守候，她足足迟了约1小时才到场。全身黑衣打扮的盈盈，神情落寞、木无表情，回答记者问题时更眼泛泪光。盈盈因为迟到，向记者说声“不好意思”后'
            '便急步入场，其助手坦言盈盈没什么可以讲。后来在《欢乐满东华2008》监制何小慧陪同下，盈盈接受简短访问，她小声地说：“多谢大家关心，交给警方处理了，不方便讲，',
            '甲醇期货今日挂牌上市继上半年焦炭、铅期货上市后，酝酿已久的甲醇期货将在今日正式挂牌交易。基准价均为3050元／吨继上半年焦炭、铅期货上市后，酝酿已久的甲醇期货将在今日正式'
            '挂牌交易。郑州商品交易所（郑商所）昨日公布首批甲醇期货8合约的上市挂牌基准价，均为3050元／吨。据此推算，买卖一手甲醇合约至少需要12200元。业内人士认为，作为国际市场上的'
            '首个甲醇期货品种，其今日挂牌后可能会因炒新资金追捧而出现冲高走势，脉冲式行情过后可能有所回落，不过，投资者在上市初期应关注期现价差异常带来的无风险套利交易机会。']
    main(test)
