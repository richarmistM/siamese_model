import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StatusDataset, SiameseStatusDataset, TripletStatusDataset
from networks import EnhancedEmbeddingNet, EnhancedSiameseNet, EnhancedTripletNet
from losses import ContrastiveLoss, OnlineTripletLoss
from trainer import fit
from utils import HardestNegativeTripletSelector
import argparse
import os


def main():

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 命令行参数配置
    parser = argparse.ArgumentParser(description='状态标签检测 - 复杂光照条件下孪生网络')

    # --- 修改 1：数据集路径使用 os.path.join 拼接 ---
    parser.add_argument('--dataset-path', type=str,
                        default=os.path.join(ROOT_DIR, 'datasets'),
                        help='数据集路径')

    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--margin', type=float, default=1.0, help='损失函数边距')
    parser.add_argument('--model-type', type=str, choices=['siamese', 'triplet'],
                        default='siamese', help='模型类型')
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(),
                        help='是否使用GPU')
    parser.add_argument('--max-samples-per-class', type=int, default=float('inf'),
                        help='每类最大样本数')

    # --- 修改 2：模型保存路径使用 os.path.join 拼接 ---
    parser.add_argument('--save-model', type=str,
                        default=os.path.join(ROOT_DIR, 'saved_models', 'siamese_model.pth'),
                        help='模型保存路径')

    parser.add_argument('--resume', type=str, default=None, help='继续训练的模型路径')

    args = parser.parse_args()rse_args()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)

    # 数据预处理 - 增强光照鲁棒性
    # ColorJitter: 随机改变亮度、对比度、饱和度，模拟复杂光照环境
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),  <-- 数字/文字通常不对称，因此不进行水平翻转
        # transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2), # 随机锐化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集（限制每类最多max_samples_per_class张）
    base_dataset = StatusDataset(args.dataset_path,
                                 transform=transform_train,
                                 max_samples_per_class=args.max_samples_per_class)
    print(f"数据集大小: {len(base_dataset)}")

    if args.model_type == 'siamese':
        # --- 使用孪生网络 (Pair-based) ---
        dataset = SiameseStatusDataset(base_dataset)
        model = EnhancedSiameseNet(EnhancedEmbeddingNet())
        loss_fn = ContrastiveLoss(args.margin)

        # 创建数据加载器
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # 验证集
        base_val_dataset = StatusDataset(args.dataset_path,
                                         transform=transform_test,
                                         max_samples_per_class=args.max_samples_per_class)
        val_dataset = SiameseStatusDataset(base_val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    elif args.model_type == 'triplet':
        # --- 使用三元组网络 (Triplet-based) ---
        dataset = TripletStatusDataset(base_dataset)
        model = EnhancedTripletNet(EnhancedEmbeddingNet())
        # 使用 Hardest Negative 策略，专注于最难区分的样本
        triplet_selector = HardestNegativeTripletSelector(args.margin)
        loss_fn = OnlineTripletLoss(args.margin, triplet_selector)

        # 创建数据加载器
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # 验证集
        base_val_dataset = StatusDataset(args.dataset_path,
                                         transform=transform_test,
                                         max_samples_per_class=args.max_samples_per_class)
        val_dataset = TripletStatusDataset(base_val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 使用GPU
    if args.cuda:
        model = model.cuda()
        if isinstance(loss_fn, OnlineTripletLoss):
            pass

    # 优化器 - 使用AdamW优化器，相比Adam有更好的Weight Decay处理，泛化性更强
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器 - 余弦退火，让学习率平滑下降
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 如果需要继续训练
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"从epoch {start_epoch}继续训练")

    # 训练
    print("开始训练...")

    # 增加 try-except 块，支持 Ctrl+C 安全退出并保存模型
    try:
        fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler,
            args.epochs, args.cuda, log_interval=10, metrics=[], start_epoch=start_epoch)
    except KeyboardInterrupt:
        print("\n\n检测到手动停止 (Ctrl+C)！正在保存当前模型，请稍候...")

    # 保存模型 (无论是否手动停止，这里都会执行)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.save_model)
    print(f"模型已保存至: {args.save_model}")


if __name__ == '__main__':
    main()