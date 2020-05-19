# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam, WarmupLinearSchedule
import pandas as pd




# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']    # bert官方将此三类免于正则化
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')    # 正无穷
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    dev_f1_score = []
    model.train()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):   # trains, labels ==>  (x, seq_len, mask), y
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss = loss / config.acc_grad
            loss.backward()
            if (i+1) % config.acc_grad == 0:   # 梯度累加
                optimizer.step()
            if total_batch % 1 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                # predic = torch.argmax(outputs.data, 1).cpu()
                train_f1 = metrics.f1_score(true, predic, average='macro')
                dev_f1, dev_loss = evaluate(config, model, dev_iter)
                dev_f1_score.append(dev_f1)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # if torch.cuda.device_count() > 1:
                    #     torch.save(model.module.state_dict(), config.save_path)  # 多gpu
                    # else:
                    torch.save(model.state_dict(), config.save_path)   # 单gp
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train F1: {2:>6.2%},  Val Loss: {3:>5.2},  Val F1: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_f1, dev_loss, dev_f1, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        print('Epoch {} Average F1-Score: {}'.format(epoch + 1, np.mean(dev_f1_score)))
        if flag or epoch == config.num_epochs-1:
            '''
                指标优化
            '''
            logits_res = []
            print('开始进行指标优化')
            for i ,(trains, lables) in enumerate(train_iter):
                logits = model(trains)
                logits = logits.data.cpu().numpy()
                lables = labels.data.cpu().numpy()
                logits_res.append(logits)
            print(lables)
            print(logits_res)
            break


    # test(config, model, test_iter)
    final_predict(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return f1, loss_total / len(data_iter), report, confusion
    return f1, loss_total / len(data_iter)


def final_predict(config, model, data_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, pred)
    # result = pd.DataFrame(predict_all)
    # result.to_csv('result.csv', index=None, encoding='utf-8')
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)
    # print('finish!!')

    return predict_all