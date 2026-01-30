import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms


class StatusDataset(Dataset):
    """
    基础状态标签数据集 (修改版)
    功能：支持自动划分训练集和验证集 (Train/Val Split)
    """

    def __init__(self, root_dir, transform=None, max_samples_per_class=float('inf'), mode='train', val_split=0.2):
        """
        :param mode: 'train' (加载80%数据) 或 'val' (加载20%数据) 或 'all' (加载所有)
        :param val_split: 验证集比例 (默认 0.2)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.mode = mode

        # 定义所有状态类别
        self.classes = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 数字
            'blank', '-', 'he', 'fen',  # 汉字
            'open', 'close',  # 英文状态
            'IndicatorLight-Bright',
            'IndicatorLight-Dark',
            'isolate_close', 'isolate_open',  # 刀闸
            'I', 'O'  # IO标识
        ]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        if self.mode == 'train':
            print(f"正在加载训练集 (保留 {(1 - val_split) * 100:.0f}%) ...")
        elif self.mode == 'val':
            print(f"正在加载验证集 (保留 {val_split * 100:.0f}%) ...")

        self.img_paths = []
        self.labels = []

        # 使用固定的随机种子，确保 'train' 和 'val' 划分是互斥且固定的
        # 这样不会出现同一张图既在训练集又在验证集的情况
        rng = np.random.RandomState(42)

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                # 1. 获取所有图片并排序 (排序是为了保证跨平台/跨次运行时顺序一致)
                all_imgs = sorted([f for f in os.listdir(class_dir)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])

                # 2. 随机打乱
                rng.shuffle(all_imgs)

                # 3. 截取最大样本数 (如果设置了 max_samples_per_class)
                if len(all_imgs) > self.max_samples_per_class:
                    all_imgs = all_imgs[:int(self.max_samples_per_class)]

                # 4. 计算分割点
                split_idx = int(len(all_imgs) * (1 - val_split))

                # 5. 根据 mode 选择图片
                if self.mode == 'train':
                    selected_imgs = all_imgs[:split_idx]
                elif self.mode == 'val':
                    selected_imgs = all_imgs[split_idx:]
                else:  # 'all'
                    selected_imgs = all_imgs

                for img_name in selected_imgs:
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"[{self.mode}] 加载完成: 共 {len(self.img_paths)} 张图片")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


class SiameseStatusDataset(Dataset):
    """
    孪生网络数据集 (Siamese Network Dataset)

    """

    def __init__(self, status_dataset, transform=None):
        self.status_dataset = status_dataset
        self.transform = transform
        # 构建标签映射
        self.labels = [status_dataset.labels[i] for i in range(len(status_dataset))]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: [i for i, lbl in enumerate(self.labels) if lbl == label]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.status_dataset[index]

        if target == 1:
            same_class_indices = self.label_to_indices[label1]
            if len(same_class_indices) > 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(same_class_indices)
                img2, _ = self.status_dataset[siamese_index]
            else:
                img2 = img1
        else:
            other_classes = [c for c in self.labels_set if c != label1]
            if other_classes:
                other_class = np.random.choice(other_classes)
                other_indices = self.label_to_indices[other_class]
                siamese_index = np.random.choice(other_indices)
                img2, _ = self.status_dataset[siamese_index]
            else:
                img2 = img1
        return (img1, img2), target

    def __len__(self):
        return len(self.status_dataset)


class TripletStatusDataset(Dataset):
    """
    三元组网络数据集 (Triplet Network Dataset)

    """

    def __init__(self, status_dataset):
        self.status_dataset = status_dataset
        # 构建标签映射
        self.labels = [status_dataset.labels[i] for i in range(len(status_dataset))]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: [i for i, lbl in enumerate(self.labels) if lbl == label]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        img1, label1 = self.status_dataset[index]
        positive_indices = self.label_to_indices[label1]
        if len(positive_indices) > 1:
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(positive_indices)
            img2, _ = self.status_dataset[positive_index]
        else:
            img2 = img1

        negative_classes = [c for c in self.labels_set if c != label1]
        if negative_classes:
            negative_class = np.random.choice(negative_classes)
            negative_indices = self.label_to_indices[negative_class]
            negative_index = np.random.choice(negative_indices)
            img3, _ = self.status_dataset[negative_index]
        else:
            img3 = img1

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.status_dataset)


class BalancedBatchSampler(BatchSampler):
    # 不需要修改
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size