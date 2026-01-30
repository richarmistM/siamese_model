import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 1. 核心特征提取网络
class EnhancedEmbeddingNet(nn.Module):
    """
    增强型嵌入网络
    基于 ResNet18，将图片转换为低维特征向量 (Embedding)。
    """
    def __init__(self, num_classes=None):
        super(EnhancedEmbeddingNet, self).__init__()

        # 加载预训练的 ResNet18 (ImageNet权重)，复用其强大的特征提取能力
        self.backbone = models.resnet18(pretrained=True)

        # 替换全连接层 (FC Layer)
        # 原始 ResNet18 输出 1000 维 (分类)，这里我们只需要输出 128 维特征
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),   # 批归一化，加速训练收敛
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),       # Dropout 防止过拟合
            nn.Linear(256, 128)    # 最终输出 128 维向量
        )

    def forward(self, x):
        x = self.backbone(x)
        # --- 关键步骤：L2 归一化 ---
        # 将输出向量映射到单位超球面上 (模长为1)。
        # 这使得 欧式距离 与 余弦相似度 具有等价性，训练更加稳定。
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_embedding(self, x):
        return self.forward(x)


# 2. 孪生网络外壳
class EnhancedSiameseNet(nn.Module):
    """
    孪生网络包装器
    接受两个输入 x1, x2，共享同一个 embedding_net 提取特征。
    """
    def __init__(self, embedding_net):
        super(EnhancedSiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


# 3. 三元组网络外壳
class EnhancedTripletNet(nn.Module):
    """
    三元组网络包装器
    接受三个输入 x1, x2, x3，共享同一个 embedding_net。
    """
    def __init__(self, embedding_net):
        super(EnhancedTripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)