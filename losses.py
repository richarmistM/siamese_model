import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    对比损失 (Contrastive Loss)
    用于 Siamese Network。
    输入：两个样本的 embedding 和 标签 target (1=同类, 0=异类)。
    逻辑：
      - 如果是同类 (target=1): 最小化它们之间的欧式距离。
      - 如果是异类 (target=0): 如果距离小于 margin，则推开它们；如果大于 margin，则不产生 loss。
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9  # 防止 sqrt(0) 导致梯度爆炸

    def forward(self, output1, output2, target, size_average=True):
        # 计算两个特征向量的欧式距离的平方
        distances = (output2 - output1).pow(2).sum(1)

        # 损失函数公式
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    三元组损失 (Triplet Loss)
    用于 Triplet Network。
    输入：Anchor, Positive, Negative 的 embeddings。
    逻辑：Loss = max(0, d(a, p) - d(a, n) + margin)
    目标：让正样本对距离 d(a,p) 至少比负样本对距离 d(a,n) 小一个 margin。
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # d(a, p)^2
        distance_negative = (anchor - negative).pow(2).sum(1)  # d(a, n)^2
        # ReLU 相当于 max(0, x)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    在线对比损失 (Online Contrastive Loss)
    特点：不预先生成 pair，而是输入一个 batch 的 embeddings，
    在 batch 内部通过 pair_selector 动态挖掘正负样本对。
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        # 动态选择成对样本
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        # 计算正样本对损失
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        # 计算负样本对损失
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    在线三元组损失 (Online Triplet Loss)
    特点：最常用的度量学习损失。在 batch 内部动态挖掘难例三元组 (Hard Negatives)。
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector  # 用于选择三元组的策略 (如 HardestNegative)

    def forward(self, embeddings, target):
        # 动态构建三元组索引 [Anchor_idx, Positive_idx, Negative_idx]
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        # 计算 d(a, p)
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        # 计算 d(a, n)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        # 计算 Loss
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)