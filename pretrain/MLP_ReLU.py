import os
import sys
sys.path.insert(0, "/data/truenas_B2/yyi/dinov2")
sys.path.insert(0, "/data/dataserver01/zhangruipeng/code/PETCT/dinov3_pretrain/dinov3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
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
from collections import defaultdict
import nibabel as nib

# ================================导入工具函数====================================
from utils import set_seed, convert_dinov3_teacher_to_hf_state_dict, preprocess_labels_and_setup_datasets
from metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate
from config import (
    MODEL_TYPE, USE_PRETRAINED, USE_CLINICAL, DEVICE, TARGET_IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    PATIENCE, UNFREEZE_LAYERS, RANDOM_SEED, NUM_FOLDS,
    TRAIN_NAME, TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH,
    IMAGE_PATH_COLUMN, LABEL_COLUMNS, TEXT_COLS,
    LOAD_LOCAL_CHECKPOINT, TEST_NAME, LOCAL_CHECKPOINT_PATH, CFG_PATH,
    IGNORE_INDEX, LOG_DIR, LOG_FILENAME
)
if MODEL_TYPE == "dinov3":
    from dinov3.models import build_model_from_cfg
    from dinov3.checkpointer import init_fsdp_model_from_checkpoint
elif MODEL_TYPE == "dinov2":
    from dinov2.models import build_model_from_cfg
# clinical
# --- 自定义 PyTorch Dataset (处理多列分类标签) ---
class MultiTaskImageDatasetFromDataFrame(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, 
                 label_cols: List[str], 
                 size: int, logger: logging.Logger,clinical_encoder, is_training: bool = False):
        self.df = df
        self.img_col = img_col
        self.label_cols = label_cols
        self.size = size
        self.logger = logger
        self.clinical_encoder = clinical_encoder
        if MODEL_TYPE in ("dinov3", "dinov2"):
            cfg = OmegaConf.load(CFG_PATH)
            self.mean = getattr(cfg.crops, "rgb_mean", [0.485, 0.456, 0.406])
            self.std  = getattr(cfg.crops, "rgb_std",  [0.229, 0.224, 0.225])
        else:
            # 非DINO模型统一使用 ImageNet 标准归一化
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
        self.processor = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean,std=self.std)
        ])

        if is_training:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]
        if str(img_path).endswith(".npy"):
            # 读取 npy
            image = np.load(img_path)  # shape: H x W 或 H x W x C
            image = np.nan_to_num(image, nan=0, posinf=1, neginf=0)
            # 如果是二维灰度图，扩展通道
            if image.max() > 1:
                image = image / 255.0

            if image.ndim == 2:
                image = np.stack([image]*3, axis=-1)

            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            # 普通图片
            image = Image.open(img_path).convert("RGB")

        
        if self.transform:
            image = self.transform(image)

        pixel_values = self.processor(image)
        if torch.isnan(pixel_values).any():
            print(f"[WARN] {img_path} 归一化后出现 NaN, mean={self.mean}, std={self.std}")

        clinical_values = self.clinical_encoder.encode(row)

        labels_dict = {}
        for task in self.label_cols:
            label_val = row[task]
            # 如果 label_val == -1（未知类别），可在此返回 None 或保留（后续 loss 忽略需特殊处理）
            labels_dict[task] = torch.tensor(label_val, dtype=torch.long)

        return pixel_values, clinical_values, labels_dict, img_path



# ====================================================================
# 2. custom_collate_fn 实现
# ====================================================================



class ClinicalEncoder:
    def __init__(self, df, text_cols):
        # 性别映射
        self.gender_map = {val: i for i, val in enumerate(df['Gender'].unique())}
        '''
        # 部位映射
        self.body_part_map = {val: i for i, val in enumerate(df['BodyPart'].unique())}'''
        # 记录维度
        self.clinical_dim = 1 + len(self.gender_map)

    def encode(self, row):
        # 1. 年龄归一化 (假设最大100岁)
        age_val = row['age']
        if pd.isna(age_val):
            age_val = 50.0 # 或者使用 df['age'].mean()
        age = torch.tensor([float(age_val) / 100.0], dtype=torch.float32)
        # 2. 性别 One-hot
        gender = torch.zeros(len(self.gender_map))
        gender_val = str(row['Gender'])
        if gender_val in self.gender_map:
            gender[self.gender_map[gender_val]] = 1.0
        '''
        # 3. 部位 One-hot
        body = torch.zeros(len(self.body_part_map))
        body_val = str(row['BodyPart'])
        if body_val in self.body_part_map:
            body[self.body_part_map[body_val]] = 1.0'''
        
        return torch.cat([age, gender])


# ---- 仅图像特征的分类头（USE_CLINICAL=False 时使用）----
class ImageOnlyHead(nn.Module):
    """不融合临床信息，直接用图像特征分类。"""
    def __init__(self, image_dim, output_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
        self.last_g = None  # 与 GatedFusionHead 接口对齐（无门控值）

    def forward(self, img_feat, cli_feat=None):
        return self.classifier(img_feat)


# ---- Gated ---
class GatedFusionHead(nn.Module):
    def __init__(self, image_dim, clinical_dim, output_dim):
        super().__init__()
        # 投影层：将不同模态对齐到同一维度
        self.img_proj = nn.Sequential(nn.Linear(image_dim, 512), nn.ReLU())
        self.cli_proj = nn.Sequential(nn.Linear(clinical_dim, 512), nn.ReLU())
        
        # 门控网络：学习一个权重来平衡两者的重要性
        self.gate = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, img_feat, cli_feat):
        i_p = self.img_proj(img_feat)
        c_p = self.cli_proj(cli_feat)
        # 计算门控值
        g = self.gate(torch.cat([i_p, c_p], dim=1))
        self.last_g = g.detach().clone()
        
        # 融合：如果g趋近1则偏向图像，趋近0则偏向临床
        fused = g * i_p + (1 - g) * c_p
        return self.classifier(fused)

# =====================================================================
# 骨干网络构建工厂函数
# =====================================================================
def build_backbone(model_type: str, logger: logging.Logger):
    """
    构建骨干网络，返回 (backbone, embed_dim)。

    支持的 model_type:
      DINO系列   : "dinov3", "dinov2"  (需要本地 checkpoint 和 CFG_PATH)
      ResNet     : "resnet18", "resnet34", "resnet50", "resnet101"
      EfficientNet: "efficientnet_b0", "efficientnet_b4"
      ViT        : "vit_b_16"
      timm       : 任意 timm 模型名称 (需要 pip install timm)
    """
    import torchvision.models as tvm

    if model_type in ("dinov3", "dinov2"):
        cfg = OmegaConf.load(CFG_PATH)
        backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
        if model_type == "dinov3":
            backbone.to_empty(device=DEVICE)
        else:
            backbone.to(device=DEVICE)

        checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        logger.info(f"加载DINO checkpoint: {LOCAL_CHECKPOINT_PATH}")

        if model_type == "dinov3":
            state_dict = checkpoint.get("teacher", checkpoint)
            model_state_dict = backbone.state_dict()
            new_state_dict = {}
            target_keys = list(model_state_dict.keys())
            for k, v in state_dict.items():
                for tk in target_keys:
                    if k.endswith(tk):
                        new_state_dict[tk] = v
                        break
        else:  # dinov2
            new_state_dict = checkpoint.get("teacher", checkpoint.get("model", checkpoint))

        msg = backbone.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Backbone loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        return backbone, embed_dim

    # ---------- ResNet ----------
    elif model_type == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
        backbone = tvm.resnet18(weights=weights)
        embed_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_type == "resnet34":
        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
        backbone = tvm.resnet34(weights=weights)
        embed_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_type == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if USE_PRETRAINED else None
        backbone = tvm.resnet50(weights=weights)
        embed_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_type == "resnet101":
        weights = tvm.ResNet101_Weights.IMAGENET1K_V2 if USE_PRETRAINED else None
        backbone = tvm.resnet101(weights=weights)
        embed_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    # ---------- EfficientNet ----------
    elif model_type == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
        backbone = tvm.efficientnet_b0(weights=weights)
        embed_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif model_type == "efficientnet_b4":
        weights = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
        backbone = tvm.efficientnet_b4(weights=weights)
        embed_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    # ---------- ViT ----------
    elif model_type == "vit_b_16":
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
        backbone = tvm.vit_b_16(weights=weights)
        embed_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()

    # ---------- timm (兜底) ----------
    else:
        try:
            import timm
            backbone = timm.create_model(model_type, pretrained=USE_PRETRAINED, num_classes=0)
            embed_dim = backbone.num_features
        except Exception as e:
            raise ValueError(
                f"不支持的模型类型: '{model_type}'。"
                f"请检查拼写，或安装 timm (pip install timm) 后使用任意 timm 模型名。\n错误: {e}"
            )

    logger.info(f"加载 {model_type} backbone，特征维度={embed_dim}，预训练={USE_PRETRAINED}")
    return backbone, embed_dim


# =====================================================================
# 特征提取统一接口
# =====================================================================
def extract_feature(backbone, pixel_values, model_type: str):
    if model_type in ("dinov3", "dinov2"):
        features = backbone.forward_features(pixel_values)
        return features["x_norm_clstoken"]
    else:
        return backbone(pixel_values)


# =====================================================================
# 统一多任务分类器（支持所有模型类型）
# =====================================================================
class MultiTaskClassifier(nn.Module):
    """
    通用多任务分类器，骨干网络由 config.MODEL_TYPE 控制。
    支持 DINO 系列、ResNet、EfficientNet、ViT 及 timm 任意模型。
    """
    def __init__(self, task_num_classes: Dict[str, int], clinical_dim, logger: logging.Logger):
        super().__init__()
        self.task_names = list(task_num_classes.keys())
        self.model_type = MODEL_TYPE

        # ========== 1. 构建骨干网络 ==========
        self.backbone, embed_dim = build_backbone(MODEL_TYPE, logger)

        # ========== 2. 冻结 / 解冻骨干层 ==========
        if UNFREEZE_LAYERS < 0:
            # 全部解冻
            for p in self.backbone.parameters():
                p.requires_grad = True
            logger.info("骨干网络全部参数已解冻")
        else:
            # 先全部冻结
            for p in self.backbone.parameters():
                p.requires_grad = False

            if UNFREEZE_LAYERS > 0:
                if MODEL_TYPE in ("dinov3", "dinov2") and hasattr(self.backbone, "blocks"):
                    # DINO Transformer blocks
                    num_layers = len(self.backbone.blocks)
                    start = max(0, num_layers - UNFREEZE_LAYERS)
                    for i in range(start, num_layers):
                        for p in self.backbone.blocks[i].parameters():
                            p.requires_grad = True
                    logger.info(f"已解冻 DINO 最后 {UNFREEZE_LAYERS} 个 Transformer Blocks")
                else:
                    # 通用方案：按子模块列表解冻最后 N 个
                    children = list(self.backbone.children())
                    n = min(UNFREEZE_LAYERS, len(children))
                    for child in children[-n:]:
                        for p in child.parameters():
                            p.requires_grad = True
                    logger.info(f"已解冻 {MODEL_TYPE} 最后 {n} 个子模块")
            else:
                logger.info("骨干网络全部参数已冻结 (UNFREEZE_LAYERS=0)")

        # ========== 3. 多任务分类头 ==========
        self.classifiers = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            out_dim = 1 if num_classes == 2 else num_classes
            if USE_CLINICAL:
                self.classifiers[task_name] = GatedFusionHead(embed_dim, clinical_dim, out_dim)
            else:
                self.classifiers[task_name] = ImageOnlyHead(embed_dim, out_dim)
        mode_str = "图像+临床融合 (GatedFusion)" if USE_CLINICAL else "仅图像特征 (ImageOnly)"
        logger.info(f"分类头模式: {mode_str}")

    def forward(self, pixel_values: torch.Tensor, clinical_values: torch.Tensor):
        pixel_values = pixel_values.to(DEVICE)
        clinical_values = clinical_values.to(DEVICE)

        if torch.isnan(pixel_values).any():
            print("pixel_values contain NaN")
        if torch.isinf(pixel_values).any():
            print("pixel_values contain Inf")
        if torch.isnan(clinical_values).any():
            print("clinical_values contain NaN")
        if torch.isinf(clinical_values).any():
            print("clinical_values contain Inf")

        if UNFREEZE_LAYERS == 0:
            with torch.no_grad():
                global_feature = extract_feature(self.backbone, pixel_values, self.model_type)
        else:
            global_feature = extract_feature(self.backbone, pixel_values, self.model_type)

        if torch.isnan(global_feature).any():
            print("global_feature contain NaN")
        if torch.isinf(global_feature).any():
            print("global_feature contain Inf")

        logits = {}
        for task_name in self.task_names:
            task_logits = self.classifiers[task_name](global_feature, clinical_values)
            if torch.isnan(task_logits).any():
                print(f"{task_name} logits contain NaN")
            if torch.isinf(task_logits).any():
                print(f"{task_name} logits contain Inf")
            logits[task_name] = task_logits

        return logits


# 向后兼容别名
DinoV3MultiTaskClassifier = MultiTaskClassifier
    
    
