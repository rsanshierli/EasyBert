import torch

class FGM():

    '''
    对于每个x:
    1.计算x的前向loss、反向传播得到梯度
    2.根据embedding矩阵的梯度计算出r，并加到当前embedding上，相当于x+r
    3.计算x+r的前向loss，反向传播得到对抗的梯度，累加到(1)的梯度上
    4.将embedding恢复为(1)时的值
    5.根据(3)的梯度对参数进行更新
    uasge:
    # 初始化
    fgm = FGM(model)
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}