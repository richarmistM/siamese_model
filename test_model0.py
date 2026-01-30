import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from datasets import StatusDataset
from networks import EnhancedEmbeddingNet, EnhancedSiameseNet
import argparse

# -------------------------- 配置参数 --------------------------
parser = argparse.ArgumentParser(description='孪生网络模型测试')
parser.add_argument('--dataset-path', type=str, default='./datasets', help='数据集根路径')
parser.add_argument('--model-path', type=str, default='./saved_models/siamese_model.pth', help='训练好的模型路径')
parser.add_argument('--test-img-path', type=str, default='./test_images/img.jpg', help='单张测试图片路径')
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(), help='是否使用GPU')
parser.add_argument('--ref-samples-per-class', type=int, default=5, help='每个类别选取的参考样本数（计算类平均嵌入）')
parser.add_argument('--test-samples-per-class', type=int, default=10, help='每个类别选取的测试样本数（计算准确率）')
parser.add_argument('--distance-type', type=str, choices=['euclidean', 'cosine'], default='cosine', help='距离计算方式')
args = parser.parse_args()


# -------------------------- 数据预处理 --------------------------
def get_transform():
    """获取测试阶段的图像预处理流程（无增强，仅标准化）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# -------------------------- 加载模型 --------------------------
def load_model(model_path, cuda):
    """加载训练好的孪生网络模型，并提取其中的 Feature Embedding 部分"""
    # 初始化模型结构
    embedding_net = EnhancedEmbeddingNet(num_classes=27)
    model = EnhancedSiameseNet(embedding_net)

    # 加载权重
    checkpoint = torch.load(model_path, map_location='cuda' if cuda else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 设置为评估模式 (关闭 Dropout, BatchNorm 使用移动平均值)
    model.eval()
    if cuda:
        model = model.cuda()
    return model.embedding_net  # 核心是嵌入网络，用于生成特征向量


# -------------------------- 距离/相似度计算 --------------------------
def calculate_distance(embedding1, embedding2, distance_type='cosine'):
    """
    计算两个嵌入向量的距离/相似度
    :param embedding1: 向量1 (shape: [1, 128])
    :param embedding2: 向量2 (shape: [N, 128])
    """
    if distance_type == 'euclidean':
        # 欧式距离：越小越相似
        return torch.norm(embedding1 - embedding2, dim=1)
    elif distance_type == 'cosine':
        # 余弦相似度：越大越相似
        return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)


# -------------------------- 生成类别参考嵌入向量 --------------------------
def generate_class_reference_embeddings(embedding_net, dataset, ref_samples_per_class, cuda):
    """
    为每个类别生成 '中心点' (Prototype)
    原理：随机取该类别的 N 个样本，提取特征后取平均值。
    """
    transform = get_transform()
    # 重新构建数据集
    base_dataset = StatusDataset(
        root_dir=dataset,
        transform=transform,
        max_samples_per_class=ref_samples_per_class
    )

    class_names = base_dataset.classes
    class_to_idx = base_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 初始化每个类别的嵌入向量列表
    class_embeddings = {cls: [] for cls in class_names}

    with torch.no_grad():
        for img_path, label in zip(base_dataset.img_paths, base_dataset.labels):
            cls_name = idx_to_class[label]
            # 加载并预处理图像
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
            if cuda:
                img_tensor = img_tensor.cuda()
            # 生成嵌入向量
            embed = embedding_net(img_tensor)
            class_embeddings[cls_name].append(embed.cpu())

    # 计算每个类别的平均嵌入向量
    class_ref_embeds = {}
    for cls_name, embeds in class_embeddings.items():
        if len(embeds) > 0:
            avg_embed = torch.mean(torch.cat(embeds, dim=0), dim=0).unsqueeze(0)  # [1, 128]
            if cuda:
                avg_embed = avg_embed.cuda()
            class_ref_embeds[cls_name] = avg_embed

    return class_ref_embeds, class_names


# -------------------------- 单张图片测试 --------------------------
def test_single_image(embedding_net, class_ref_embeds, img_path, cuda, distance_type):
    """
    测试单张图片，返回最匹配的类别。
    使用 KNN (k=1) 思想：将测试图与所有类别的中心点比对，距离最近者即为预测类别。
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"测试图片不存在: {img_path}")

    # 预处理图片
    transform = get_transform()
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    if cuda:
        img_tensor = img_tensor.cuda()

    # 生成嵌入向量
    with torch.no_grad():
        test_embed = embedding_net(img_tensor)

    # 计算与每个类别参考向量的距离/相似度
    cls_scores = {}
    for cls_name, ref_embed in class_ref_embeds.items():
        score = calculate_distance(test_embed, ref_embed, distance_type).item()
        cls_scores[cls_name] = score

    # 确定最匹配的类别
    if distance_type == 'euclidean':
        best_cls = min(cls_scores, key=cls_scores.get)
        best_score = cls_scores[best_cls]
        print(f"【单张图片测试】\n图片路径: {img_path}\n最匹配类别: {best_cls}\n欧式距离: {best_score:.4f}")
    else:  # cosine
        best_cls = max(cls_scores, key=cls_scores.get)
        best_score = cls_scores[best_cls]
        print(f"【单张图片测试】\n图片路径: {img_path}\n最匹配类别: {best_cls}\n余弦相似度: {best_score:.4f}")

    # 可选：打印所有类别的得分
    print("\n所有类别得分（按匹配度降序）:")
    sorted_cls = sorted(cls_scores.items(), key=lambda x: x[1], reverse=(distance_type == 'cosine'))
    for cls, score in sorted_cls[:5]:
        print(f"  {cls}: {score:.4f}")

    return best_cls


# -------------------------- 数据集批量测试 --------------------------
def test_dataset_accuracy(embedding_net, dataset_path, class_ref_embeds, test_samples_per_class, cuda, distance_type):
    """
    从数据集每个类别随机选样本测试，计算整体准确率
    """
    transform = get_transform()
    base_dataset = StatusDataset(
        root_dir=dataset_path,
        transform=transform,
        max_samples_per_class=test_samples_per_class
    )

    idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
    correct = 0
    total = 0

    print("\n" + "-" * 50)
    print(f"【数据集批量测试】每个类别选取 {test_samples_per_class} 个样本")

    with torch.no_grad():
        for img_path, label in zip(base_dataset.img_paths, base_dataset.labels):
            true_cls = idx_to_class[label]
            total += 1

            # 预处理图片并生成嵌入
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            if cuda:
                img_tensor = img_tensor.cuda()
            test_embed = embedding_net(img_tensor)

            # 计算与所有类别参考向量的距离/相似度
            cls_scores = {}
            for cls_name, ref_embed in class_ref_embeds.items():
                score = calculate_distance(test_embed, ref_embed, distance_type).item()
                cls_scores[cls_name] = score

            # 确定预测类别
            if distance_type == 'euclidean':
                pred_cls = min(cls_scores, key=cls_scores.get)
            else:
                pred_cls = max(cls_scores, key=cls_scores.get)

            # 统计正确数
            if pred_cls == true_cls:
                correct += 1

            # 每10个样本打印一次进度
            if total % 10 == 0:
                print(f"已测试 {total} 个样本，当前准确率: {correct / total:.4f}")

    # 计算最终准确率
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n【批量测试结果】总测试样本数: {total} | 正确数: {correct} | 准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


# -------------------------- 主函数 --------------------------
if __name__ == '__main__':
    # 1. 加载模型
    print("加载模型...")
    embedding_net = load_model(args.model_path, args.cuda)

    # 2. 生成类别参考嵌入向量
    print("生成类别参考嵌入向量...")
    class_ref_embeds, class_names = generate_class_reference_embeddings(
        embedding_net, args.dataset_path, args.ref_samples_per_class, args.cuda
    )

    # 3. 单张图片测试
    print("\n" + "-" * 50)
    try:
        test_single_image(
            embedding_net, class_ref_embeds, args.test_img_path, args.cuda, args.distance_type
        )
    except Exception as e:
        print(f"单张图片测试失败: {e}")

    # 4. 数据集批量测试（计算准确率）(训练集）
    test_dataset_accuracy(
        embedding_net, args.dataset_path, class_ref_embeds, args.test_samples_per_class, args.cuda, args.distance_type
    )