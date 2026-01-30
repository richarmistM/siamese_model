import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import matplotlib.cm as cm

# 引入你的项目模块
from datasets import StatusDataset
from networks import EnhancedEmbeddingNet, EnhancedSiameseNet

# -------------------------- 配置参数 --------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='模型评估与可视化')

# --- 修改 1：数据集路径 ---
parser.add_argument('--dataset-path', type=str,
                    default=os.path.join(ROOT_DIR, 'datasets'),
                    help='数据集路径')

# --- 修改 2：模型加载路径 ---
parser.add_argument('--model-path', type=str,
                    default=os.path.join(ROOT_DIR, 'saved_models', 'siamese_model.pth'),
                    help='模型路径')

parser.add_argument('--batch-size', type=int, default=64, help='提取特征时的批量大小')
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(), help='是否使用GPU')
parser.add_argument('--plot-method', type=str, default='pca', choices=['pca', 'tsne'],
                    help='可视化降维方法 (pca 或 tsne)')
args = parser.parse_args()


# -------------------------- 工具函数 --------------------------
def get_transform():
    """测试/验证用的预处理（仅标准化）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
#

def load_embedding_model(model_path, cuda):
    """加载模型并返回特征提取部分"""
    # 这里的 num_classes 仅用于初始化结构，不影响权重加载
    embedding_net = EnhancedEmbeddingNet(num_classes=27)
    model = EnhancedSiameseNet(embedding_net)

    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cuda' if cuda else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    if cuda:
        model = model.cuda()
    return model.embedding_net


def extract_embeddings(dataloader, model, cuda):
    """
    遍历 DataLoader，提取所有图片的 Embedding 和 Label
    """
    embeddings = []
    labels = []

    print(f"正在提取特征 (共 {len(dataloader.dataset)} 张图片)...")

    with torch.no_grad():
        for imgs, target in dataloader:
            if cuda:
                imgs = imgs.cuda()

            # 提取特征 [Batch, 128]
            emb = model(imgs)

            embeddings.append(emb.cpu().numpy())
            labels.append(target.numpy())

    # 拼接成大矩阵
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels


def plot_embeddings(embeddings, labels, classes, title, save_name, method='pca'):
    """
    绘制二维散点图
    """
    print(f"正在生成可视化图表: {title} ({method})...")

    # 1. 降维 (128维 -> 2维)
    if method == 'tsne':
        # t-SNE 效果更好看，但速度慢，适合数据量 < 10000
        reducer = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    else:
        # PCA 速度快，适合观察整体分布
        reducer = PCA(n_components=2)

    embeddings_2d = reducer.fit_transform(embeddings)

    # 2. 绘图
    plt.figure(figsize=(12, 10))

    # 哪怕有20多个类别，也尽量生成区分度高的颜色
    unique_labels = np.unique(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label_idx in enumerate(unique_labels):
        # 找到当前类别的所有点
        idxs = labels == label_idx
        class_name = classes[label_idx]

        plt.scatter(embeddings_2d[idxs, 0],
                    embeddings_2d[idxs, 1],
                    color=colors[i],
                    label=class_name,
                    alpha=0.6,
                    s=20)  # alpha是透明度，s是点的大小

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(title)
    plt.tight_layout()

    # 保存
    plt.savefig(save_name, dpi=300)
    print(f"图表已保存至: {save_name}")
    # plt.show() # 如果在服务器运行，请注释掉此行


def calculate_accuracy(embeddings, labels, prototypes, distance_type='cosine'):
    """
    计算准确率：将每个样本的 Embedding 与 类别中心(Prototypes) 进行比对
    """
    correct = 0
    total = len(labels)

    # 将 prototypes 转换为矩阵方便计算 [Num_Classes, 128]
    # prototypes 是一个 dict: {label_idx: embedding_tensor}
    proto_indices = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[k] for k in proto_indices])  # [C, 128]

    # 将 embeddings 转为 Tensor [N, 128]
    embed_tensor = torch.from_numpy(embeddings)
    label_tensor = torch.from_numpy(labels)

    if distance_type == 'cosine':
        # 余弦相似度计算: A . B^T
        # 假设都已经归一化了(模型里有L2 Normalize)，直接点积即可
        # [N, 128] x [128, C] -> [N, C]
        scores = torch.mm(embed_tensor, proto_matrix.t())
        # 取分数最大的索引 (即最相似的类别在 proto_indices 中的位置)
        preds_indices = scores.argmax(dim=1)
        # 映射回真实的 label_idx
        preds = torch.tensor([proto_indices[i] for i in preds_indices])

    else:  # euclidean
        # 欧式距离计算
        # (x-y)^2 = x^2 + y^2 - 2xy
        # 这里为了简单，我们用 torch.cdist [N, C]
        dists = torch.cdist(embed_tensor, proto_matrix)
        # 取距离最小的
        preds_indices = dists.argmin(dim=1)
        preds = torch.tensor([proto_indices[i] for i in preds_indices])

    correct = preds.eq(label_tensor).sum().item()
    return correct / total


def get_prototypes(embeddings, labels):
    """
    计算每个类别的平均特征向量 (即 Prototype / Center)
    用于分类参考
    """
    unique_labels = np.unique(labels)
    prototypes = {}

    for label in unique_labels:
        # 取出该类别的所有向量
        class_embeddings = embeddings[labels == label]
        # 求平均
        mean_embedding = np.mean(class_embeddings, axis=0)
        # 转为 Tensor 并归一化 (保持和模型输出一致)
        mean_embedding = torch.from_numpy(mean_embedding)
        # 再次 L2 归一化，确保它在单位球面上
        mean_embedding = torch.nn.functional.normalize(mean_embedding.unsqueeze(0), p=2).squeeze(0)
        prototypes[label] = mean_embedding

    return prototypes


# -------------------------- 主流程 --------------------------
def main():
    # 1. 准备数据和模型
    transform = get_transform()
    model = load_embedding_model(args.model_path, args.cuda)

    # 加载训练集 (mode='train')
    train_dataset = StatusDataset(args.dataset_path, transform=transform, mode='train', val_split=0.2)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载验证集 (mode='val')
    val_dataset = StatusDataset(args.dataset_path, transform=transform, mode='val', val_split=0.2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 提取特征
    print("\n" + "=" * 50)
    print("STEP 1: 处理训练集 (Train Set)")
    train_embeds, train_labels = extract_embeddings(train_loader, model, args.cuda)

    print("\n" + "=" * 50)
    print("STEP 2: 处理验证集 (Validation Set)")
    val_embeds, val_labels = extract_embeddings(val_loader, model, args.cuda)

    # 3. 计算类别中心 (Prototypes) - 必须仅使用训练集计算！
    print("\n" + "=" * 50)
    print("STEP 3: 计算基准 (Class Prototypes)")
    # 使用训练集的数据来定义“每个类别长什么样”
    prototypes = get_prototypes(train_embeds, train_labels)
    print(f"已计算 {len(prototypes)} 个类别的中心向量")

    # 4. 计算准确率
    print("\n" + "=" * 50)
    print("STEP 4: 评估准确率 (Accuracy)")

    # 训练集准确率 (Self-Check: 应该非常高，接近 100%)
    train_acc = calculate_accuracy(train_embeds, train_labels, prototypes)
    print(f"【训练集准确率】: {train_acc:.4f} ({train_acc * 100:.2f}%)")

    # 验证集准确率 (Real Performance: 这是模型真正的能力)
    val_acc = calculate_accuracy(val_embeds, val_labels, prototypes)
    print(f"【验证集准确率】: {val_acc:.4f} ({val_acc * 100:.2f}%)")

    # 5. 可视化绘图
    print("\n" + "=" * 50)
    print("STEP 5: 生成散点图 (Visualization)")

    # 获取类别名称列表用于图例
    class_names = train_dataset.classes

    # 绘制训练集分布
    plot_embeddings(train_embeds, train_labels, class_names,
                    title=f"Train Set Embeddings (Acc: {train_acc:.2%})",
                    save_name="embeddings_train.png",
                    method=args.plot_method)

    # 绘制验证集分布
    plot_embeddings(val_embeds, val_labels, class_names,
                    title=f"Validation Set Embeddings (Acc: {val_acc:.2%})",
                    save_name="embeddings_val.png",
                    method=args.plot_method)

    print("\n所有工作已完成！请查看生成的 embeddings_train.png 和 embeddings_val.png")


if __name__ == '__main__':
    main()