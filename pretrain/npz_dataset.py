import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Dataset Class ---
class NPZDataset(Dataset):
    def __init__(self, data_dir, image_key='img', global_crops_scale=(0.75, 1.0), local_crops_scale=(0.05, 0.4),global_crops_size=518, local_crops_size=112, local_crops_number=8):
        self.data_dir = data_dir
        self.image_key = image_key
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.local_crops_number = local_crops_number
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop((global_crops_size, global_crops_size), scale=global_crops_scale),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop((local_crops_size, local_crops_size), scale=local_crops_scale),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.loader = lambda path: np.load(path)[self.image_key]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        image_np = self.loader(path)
        if image_np.dtype == np.uint16 or image_np.dtype == np.int16:
            max_val = np.max(image_np)
            if max_val > 0:
                # 归一化到 0-255
                image_8bit = (image_np / max_val * 255).astype(np.uint8)
            else:
                image_8bit = image_np.astype(np.uint8)
        from PIL import Image
        image_pil = Image.fromarray(image_np.astype('uint8'))

        global_crops = [self.global_transform(image_pil) for _ in range(2)]
        local_crops = [self.local_transform(image_pil) for _ in range(self.local_crops_number)]

        # 返回一个元组，其中第一个元素是包含裁切的字典
        sample = {
            "global_crops": global_crops,
            "local_crops": local_crops
        }
        # 官方数据加载器通常还返回一个“目标”或“标签”，即使在自监督任务中它可能只是一个占位符。
        # 我们可以简单地返回一个 None 或 0。
        return (sample, None)