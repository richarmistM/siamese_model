import numpy as np


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    累计准确率指标
    通常用于分类任务，但在度量学习中可能较少直接使用，除非加上分类头。
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    """
    平均非零三元组数量指标
    意义：统计一个 Batch 中有多少个三元组产生了 Loss (Loss > 0)。
    如果这个数值随着训练急剧下降，说明大部分三元组都已经被模型“解决”了（距离拉开了）。
    如果数值一直很高，说明训练比较困难。
    """

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        # loss[1] 通常在 OnlineTripletLoss 中返回的是有效三元组的数量
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'