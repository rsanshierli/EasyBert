class EMA():
    '''
    权重滑动平均，对最近的数据给予更高的权重
    uasge：
    # 初始化
    ema = EMA(model, 0.999)
    ema.register()

    # 训练过程中，更新完参数后，同步update shadow weights
    def train():
        optimizer.step()
        ema.update()

    # eval前，apply shadow weights；
    # eval之后（保存模型后），恢复原来模型的参数
    def evaluate():
        ema.apply_shadow()
        # evaluate
        ema.restore()
    '''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
