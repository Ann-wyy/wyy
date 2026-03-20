import os
import random
import sys
sys.path.insert(0, "/data/truenas_B2/yyi/dinov2")
sys.path.insert(0, "/data/truenas_B2/yyi/dinov3_pretrain")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from dinov3.models import build_model_from_cfg
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import default_collate
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ================================导入工具函数====================================
from metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate
 
# --- 配置参数 ---
TARGET_IMAGE_SIZE = 256 # 图像目标尺寸
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PATIENCE = 10 # 早停耐心值
RANDOM_SEED = 42 # 42, 100, 601, 1010, 2025


# 自动选择 GPU 设备，优先使用 cuda:0
DEVICE = "cuda:0"

# 用户提供的文件路径
MODEL_TYPE = "vit_base"

TRAIN_NAME = f"BTXRD"
CSV_PATH = "/home/yyi/data/test_dataset/BTXRD_dataset.csv" # 标签CSV文件路径
IMAGE_PATH_COLUMN = 'image_path' # CSV中包含图像相对路径的列名
LABEL_COLUMNS = ['tumor','benign','malignant']  # 您的所有标签列名
LOCAL_CHECKPOINT_PATH = "/data/truenas_B2/yyi/weight/dinov3_vitb16_pretrain_teacher.pth" # 替换为您的本地 .pth 文件路径
CFG_PATH = "/home/yyi/CODE/model/dinov3_vitb16_pretrain.yaml"
LOAD_LOCAL_CHECKPOINT = True# 是否加载本地检查点
if LOAD_LOCAL_CHECKPOINT:
    TEST_NAME = "Dinov3_vitb"
else:
    TEST_NAME = "Dinov3"
TEST_NAME = f"{TEST_NAME}_{TRAIN_NAME}_{TARGET_IMAGE_SIZE}_{LEARNING_RATE}_{RANDOM_SEED}"
IGNORE_INDEX = -1
# **新增：日志配置函数**
LOG_DIR = f"/data/truenas_B2/yyi/logs/{TRAIN_NAME}/{TEST_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"{TEST_NAME}_{time.strftime('%Y%m%d-%H%M%S')}.log")

def set_seed(seed):
    """设置所有必要的随机种子"""
    # Python 内建的随机数
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # GPU (CUDA) 种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 强制 CUDA 禁用非确定性算法，确保结果完全一致
        # 但可能会轻微降低一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED) # 设置随机种子

def setup_logging():
    """配置日志记录，输出到文件和控制台。"""
    if logging.getLogger().hasHandlers():
        return logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(LOG_FILENAME), # 写入文件
            logging.StreamHandler() # 输出到控制台
        ]
    )
    return logging.getLogger(__name__)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
logger = setup_logging() # 初始化全局日志记录器
logger.info(f"随机种子: {RANDOM_SEED}")

from collections import OrderedDict
class MultiTaskImageDatasetFromDataFrame(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, label_cols: List[str], 
                 size: int, logger: logging.Logger, is_training: bool = False):
        self.df = df
        self.img_col = img_col
        self.label_cols = label_cols
        self.size = size
        self.logger = logger
        cfg = OmegaConf.load(CFG_PATH)
        mean = getattr(cfg.crops, "rgb_mean", None) 
        std  = getattr(cfg.crops, "rgb_std", None)
        self.in_chans = getattr(cfg.teacher, "in_chans", None)  

        if mean is None or std is None:
            raise ValueError("config 里缺少 rgb_mean / rgb_std")
        mean = list(mean)
        std = list(std)

        self.processor = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(mean=mean,std=std)
        ])

        if is_training:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=20),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]
        if self.in_chans == 3:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                self.logger.warning(f"图像损坏或无法加载: {img_path}")
                return None, None, img_path
        else:
            try:
                image = Image.open(img_path).convert("L")
            except Exception as e:
                self.logger.warning(f"图像损坏或无法加载: {img_path}")
                return None, None, img_path
        
        if self.transform:
            image = self.transform(image)

        pixel_values = self.processor(image)


        labels_dict = {}
        for task in self.label_cols:
            label_val = row[task]
            # 如果 label_val == -1（未知类别），可在此返回 None 或保留（后续 loss 忽略需特殊处理）
            labels_dict[task] = torch.tensor(label_val, dtype=torch.long)

        return pixel_values, labels_dict, img_path



# ====================================================================
# 2. custom_collate_fn 实现
# ====================================================================
# ================== 改进 custom_collate_fn ==================
def custom_collate_fn(batch: List[Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
    # 过滤掉损坏图像
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        # 返回空 tensor，防止 DataLoader 报错
        return torch.empty(0), {}, []

    pixel_values = torch.stack([item[0] for item in batch])
    
    task_names = list(batch[0][1].keys())
    labels_dict = {}
    for task_name in task_names:
        labels = [item[1][task_name] for item in batch]
        labels_dict[task_name] = torch.stack(labels)  # shape: [N]

    img_paths = [item[2] for item in batch]
    return pixel_values, labels_dict, img_paths


# --- 自定义模型：DINOv3 + 多个分类头 ---

class DinoV3MultiTaskClassifier(nn.Module):
    """
    基于 DINOv3 主干网络，带有多任务分类头。
    """
    def __init__(self, model_name: str, task_num_classes: Dict[str, int]):
        super().__init__()

        self.task_names = list(task_num_classes.keys())
        cfg = OmegaConf.load(CFG_PATH)
        self.backbone, self.embed_dim = build_model_from_cfg(cfg, only_teacher=True)
        self.backbone.to_empty(device=DEVICE)
        checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location=DEVICE)
        logger.info(f"load checkpoint: {LOCAL_CHECKPOINT_PATH}")
        state_dict = checkpoint.get("teacher", checkpoint)
        new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone" in k}
        msg = self.backbone.load_state_dict(new_state_dict, strict=False)
        logger.info(
            f"Backbone loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
        )
        # 冻结主干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 定义多个分类头
        self.classifiers = nn.ModuleDict()
        feature_dim = self.embed_dim
        # feature_dim = self.embed_dim * 2
        for task_name, num_classes in task_num_classes.items():
            if num_classes == 2:
                output_dim = 1
            elif num_classes > 2:
                output_dim = num_classes
            else:
                logger.warning(f"任务 '{task_name}' 的类别数 {num_classes} 无效。设置为 1。")
                output_dim = 1
            self.classifiers[task_name] = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(feature_dim // 2, output_dim)
            )

        # 确保分类头参数是可训练的
        for name in self.classifiers:
            for p in self.classifiers[name].parameters():
                p.requires_grad = True

        

        
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 运行主干网络（冻结）
        pixel_values = pixel_values.to(DEVICE)
        
        # 即使主干网络冻结，也要确保它在正确的设备上运行
        tokens = self.backbone.get_intermediate_layers(pixel_values, n=1)[0]

        cls_token = tokens[:, 0, :]        # CLS token [B, D]
        patch_mean = tokens[:, 1:, :].mean(dim=1)  # patch 平均 [B, D]
        # global_feature = torch.cat([cls_token, patch_mean], dim=1)
        global_feature = (cls_token + patch_mean) / 2
        # global_feature = cls_token

        # 运行各个分类头
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.classifiers[task_name](global_feature)

        return logits
    



# --- 训练函数 (新增日志和早停逻辑) ---
def train_multi_task_classifier(logger: logging.Logger):
    # --- TENSORBOARD 初始化 ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    logger.info(f"TensorBoard Writer initialized at: {LOG_DIR}")
    best_model_path = os.path.join(LOG_DIR, "best_model.pth")

    # 读取数据集
    df = pd.read_csv(CSV_PATH)
    stratify_col = df[LABEL_COLUMNS].astype(str).agg('_'.join, axis=1)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=stratify_col  # stratify 保持类别分布
    )

    temp_stratify_col = temp_df[LABEL_COLUMNS].astype(str).agg('_'.join, axis=1)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_stratify_col
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # 对每个任务列做编码
    num_classes_dict = {}
    for task in LABEL_COLUMNS:
        le = LabelEncoder()
        # 只编码非 -1 的值
        mask = train_df[task] != -1
        train_df.loc[mask, task] = le.fit_transform(train_df.loc[mask, task])
        
        # 同样转换 val/test
        val_df.loc[val_df[task] != -1, task] = le.transform(val_df.loc[val_df[task] != -1, task])
        test_df.loc[test_df[task] != -1, task] = le.transform(test_df.loc[test_df[task] != -1, task])

        num_classes_dict[task] = len(le.classes_)


    logger.info(f"数据集已加载 -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    
    def create_dataset(df, is_train=False):
        return MultiTaskImageDatasetFromDataFrame(
            df=df,
            img_col=IMAGE_PATH_COLUMN,
            label_cols=LABEL_COLUMNS,
            size=TARGET_IMAGE_SIZE,
            logger=logger,
            is_training=is_train
        )
    
    train_dataset = create_dataset(train_df, is_train=True)
    val_dataset = create_dataset(val_df, is_train=False)
    test_dataset = create_dataset(test_df, is_train=False)  # 用于最终测试

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)

    # --- 类别不平衡处理与标签反转 & 权重计算 ---
    task_weights = {}

    for task in LABEL_COLUMNS:
        num_cls = num_classes_dict[task]
        labels = train_df[task].values
        valid_labels = labels[labels != -1]

        # --- 二分类标签反转逻辑 ---
        if num_cls == 2:
            counts = pd.Series(valid_labels).value_counts()
            if len(counts) == 2:
                minority_encoded_val = counts.idxmin()  # 少数类编码
                if minority_encoded_val == 0:
                    logger.warning(f"任务 '{task}' 的少数类被编码为 0，进行全局反转...")
                    label_mapping = {0: 1, 1: 0, -1: -1}
                    for df_ in [train_df, val_df, test_df]:
                        df_[task] = df_[task].map(label_mapping)
                    valid_labels = train_df[task].values[train_df[task] != -1]

        # --- 警告训练集中未覆盖全部类别 ---
        if len(np.unique(valid_labels)) < num_cls:
            logger.warning(f"训练集中任务 '{task}' 未覆盖所有类别")

        # --- 生成 task_weights ---
        if num_cls == 2:
            counts = np.bincount(valid_labels.astype(int))
            # 防止某类数量为0
            counts = np.where(counts == 0, 1, counts)
            # 对二分类 pos_weight = 负类样本数 / 正类样本数
            weight_tensor = torch.tensor(counts, dtype=torch.float32, device=DEVICE)
        else:
            counts = np.bincount(valid_labels.astype(int))
            counts = np.where(counts == 0, 1, counts)
            # 多分类使用 max/count 做 class weight
            weight_tensor = torch.tensor(counts.max() / counts, dtype=torch.float32, device=DEVICE)

        task_weights[task] = weight_tensor
        logger.info(f"任务 '{task}' 权重: {task_weights[task].cpu().numpy()}")

    # --- 创建任务特定的加权损失函数字典 ---
    criterion_dict = {}
    for task in LABEL_COLUMNS:
        weight = task_weights[task]
        num_cls = num_classes_dict[task]

        if num_cls == 2:
            # --- 二分类 BCEWithLogitsLoss ---
            w_neg = weight[0].item()
            w_pos = weight[1].item()
            # pos_weight = 负类样本数 / 正类样本数
            pos_weight_scalar = torch.tensor(w_neg / max(w_pos, 1), dtype=torch.float32, device=DEVICE)
            logger.info(f"任务 '{task}' 的 BCEWithLogitsLoss 使用 pos_weight={pos_weight_scalar.item():.4f}")
            criterion_dict[task] = nn.BCEWithLogitsLoss(pos_weight=pos_weight_scalar)

        elif num_cls > 2:
            criterion_dict[task] = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
            logger.info(f"任务 '{task}' 的 CrossEntropyLoss 使用 class weights: {weight.cpu().numpy()}")
        else:
            logger.error(f"任务 '{task}' 的类别数 {num_cls} 无效。使用默认 CrossEntropyLoss。")
            criterion_dict[task] = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    model = DinoV3MultiTaskClassifier(model_name=MODEL_TYPE, task_num_classes=num_classes_dict)
    model.to(DEVICE)
    # 仅优化分类头参数 (假设主干网络冻结)
    optimizer = torch.optim.AdamW(model.classifiers.parameters(), lr=LEARNING_RATE)

    # 初始化 GradScaler
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',                 # 监控分数，所以使用 'max'
        factor=0.5,                 # 每次降低到原来的 50%，避免一下子降太猛
        patience=5,                 # 比早停耐心值小，给低 LR 精调留空间
        min_lr=1e-6                 # 足够低，允许充分精调
    )
    logger.info(f"学习率调度器 ReduceLROnPlateau 已初始化，监控模式: max, 降低耐心值: 5")

    logger.info(f"模型已加载，在设备 {DEVICE} 上训练...")

    best_val_score = -1.0
    patience_counter = 0

    best_epoch = -1

    # 4. 训练循环
    task_names = LABEL_COLUMNS
    for epoch in range(NUM_EPOCHS):
        total_combined_loss = 0
        train_labels_all = {task: [] for task in task_names}
        train_probs_all = {task: [] for task in task_names}
        train_paths_all = []
        model.train()

        # 训练步骤
        for step, batch in enumerate(train_loader):
            if batch is None:
                logger.warning("Received an empty batch after filtering corrupt files. Skipping step.")
                continue
            pixel_values, labels_dict, img_paths = batch
            batch_size = pixel_values.size(0)
            pixel_values = pixel_values.to(DEVICE)
            for task in labels_dict:
                labels_dict[task] = labels_dict[task].to(DEVICE)

            optimizer.zero_grad()
            combined_loss = torch.tensor(0.0, device=DEVICE)
            train_paths_all.extend(img_paths)

            with torch.amp.autocast(device_type='cuda'):
                predictions_dict = model(pixel_values)
                for task_name in model.task_names:
                    logits = predictions_dict[task_name]  # shape: (batch_size, 1)
                    labels = labels_dict[task_name]   
                    num_cls = num_classes_dict[task_name]         # shape: (batch_size)
                    task_criterion = criterion_dict[task_name]
                    task_loss = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)

                    valid_mask = (labels != -1)
                    valid_count = valid_mask.sum()
                    if valid_count == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask].long()      # [N_valid]

                    if num_cls == 2:  
                        # BCEWithLogitsLoss 需要浮点型的 target，形状为 (N, 1)
                        target = valid_labels.float().unsqueeze(1)  # [N,1]
                        valid_logits = valid_logits.view(-1,1)

                        
                        task_loss = task_criterion(valid_logits, target)
                        # 计算全部样本的概率 (用于指标统计)，需用完整 logits 保持与 labels 长度一致
                        probs_pos = torch.sigmoid(logits).squeeze(1)
                        probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1)
                    else:
                        target = valid_labels.long() 
                        task_loss = task_criterion(valid_logits , target)
                        probabilities = torch.softmax(logits , dim=1) 

                    combined_loss += task_loss # 累加总损失
                    train_probs_all[task_name].extend(probabilities.cpu().tolist())
                    train_labels_all[task_name].extend(labels.cpu().tolist())

            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_combined_loss += combined_loss.item()
            
            # 记录迭代训练损失
            if step % 50 == 0 and step > 0:
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Step {step}/{len(train_loader)}, "
                        f"Total Train Loss: {combined_loss.item():.4f}")

        # --- 训练评估
        logger.info(f"--------- Epoch {epoch + 1} 训练评估总结 --------")
        train_metrics = {}
        for task_name in task_names:
            num_cls = num_classes_dict[task_name]
            metrics = calculate_metrics(
                all_labels=train_labels_all[task_name],
                all_probs=train_probs_all[task_name],
                num_classes=num_cls,
                task_name=task_name,
                mode=f'train_{task_name}',
                logger=logger
            )
            train_metrics[task_name] = metrics

        log_metrics_to_tensorboard(
            writer, 
            train_metrics, 
            epoch + 1, 
            'Train', 
            logger
        )

        # --- Epoch 结束后的评估 ---
        logger.info(f"--------- Epoch {epoch + 1} 验证评估总结 --------")
        val_metrics = evaluate(
            model, val_loader, criterion_dict, model.task_names, num_classes_dict, DEVICE, mode='val', logger=logger
        )
        log_metrics_to_tensorboard(writer, val_metrics, epoch + 1, 'Val', logger)
        # 用所有任务的平均 AUROC 作为早停指标，避免只盯着一个任务
        auroc_values = [val_metrics[t]['auroc'] for t in model.task_names if val_metrics[t]['auroc'] > 0]
        val_score = float(np.mean(auroc_values)) if auroc_values else 0.0
        logger.info(f"所有任务平均 Val AUROC: {val_score:.4f}")
        # === 学习率调度器步进 ===
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率 (LR): {current_lr:.2e}")
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_epoch = epoch + 1
            logger.info(f"最佳模型auroc分数: {best_val_score:.4f}")
            try:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_score': best_val_score,
                    'task_names': task_names
                }, best_model_path)
                logger.info(f"✅ 模型权重已保存到: {best_model_path}")
            except Exception as e:
                logger.error(f"保存模型权重失败: {e}")
        else:
            patience_counter += 1
            logger.info(f"🖤验证未改善。当前耐心值: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info(f"🛑 早停触发！在 Epoch {epoch + 1} 停止训练。")
                break

    logger.info("\n多任务训练完成！")
    # 加载最佳模型权重
    if os.path.exists(best_model_path):
        logger.info(f"正在从 {best_model_path} 加载最佳模型权重...")
        try:
            # 💥 关键修改：加载最佳模型
            checkpoint = torch.load(best_model_path, map_location=DEVICE,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            # 确保模型处于评估模式
            model.eval() 
            logger.info(f"模型已成功加载 (最佳 Epoch: {checkpoint['epoch']}, Score: {checkpoint['best_val_score']:.4f})")
        except Exception as e:
            logger.critical(f"加载最佳模型失败: {e}")
            return # 如果加载失败，则无法进行测试评估
    else:
        logger.warning("未找到最佳模型检查点，使用当前模型状态进行测试评估。")
        model.eval() # 切换到评估模式
    test_metrics = evaluate(model, test_loader, criterion_dict, model.task_names, num_classes_dict, DEVICE, mode='test', logger=logger)
    writer.close() # 确保所有数据写入日志文件
    return None


if __name__ == "__main__":
    # 初始化日志记录器
    main_logger = setup_logging()
    main_logger.info(f"日志文件已创建：{LOG_FILENAME}")
    main_logger.info(f"运行设备: {DEVICE}")
    main_logger.info(f"图像尺寸: {TARGET_IMAGE_SIZE}")
    main_logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    main_logger.info(f"LEARNING_RATE: {LEARNING_RATE}")

    trained_model = train_multi_task_classifier(main_logger)

    if trained_model:
        main_logger.info("\n最终模型已训练并加载。")
