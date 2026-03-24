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
import random
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# ================================导入工具函数====================================
from metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate
from MLP_ReLU import MultiTaskImageDatasetFromDataFrame, ClinicalEncoder, MultiTaskClassifier, DinoV3MultiTaskClassifier
from config import (
    MODEL_TYPE, USE_PRETRAINED, USE_CLINICAL, DEVICE, TARGET_IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    PATIENCE, UNFREEZE_LAYERS, RANDOM_SEED, NUM_FOLDS,
    TRAIN_NAME, TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH,
    IMAGE_PATH_COLUMN, LABEL_COLUMNS, TEXT_COLS,
    LOAD_LOCAL_CHECKPOINT, TEST_NAME, LOCAL_CHECKPOINT_PATH, CFG_PATH,
    IGNORE_INDEX, LOG_DIR, LOG_FILENAME
)
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


def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None

    pixel_values = torch.stack([item[0] for item in batch])
    clinical_values = torch.stack([item[1] for item in batch]) # 新增
    if torch.isnan(pixel_values).any():
        print("[WARN] Collate pixel_values 出现 NaN")
    if torch.isnan(clinical_values).any():
        print("[WARN] Collate clinical_values 出现 NaN")
    
    # 保持 labels 处理逻辑不变
    task_names = list(batch[0][2].keys())
    labels_dict = {name: torch.stack([item[2][name] for item in batch]) for name in task_names}
    img_paths = [item[3] for item in batch]
    
    return pixel_values, clinical_values, labels_dict, img_paths

def save_sample_analysis(correct_record, prob_record, save_path):

    rows = []

    all_paths = set(correct_record.keys()) | set(prob_record.keys())

    for path in all_paths:

        rows.append({
            "image_path": path,
            "ever_correct": correct_record.get(path, 0),
            "max_probability": prob_record.get(path, 0.0)
        })

    df = pd.DataFrame(rows)

    df.to_csv(save_path, index=False)


# --- 训练函数 (新增日志和早停逻辑) ---
def train_multi_task_classifier(logger: logging.Logger):
    global IMAGE_PATH_COLUMN, LABEL_COLUMNS, TRAIN_CSV_PATH
    writer = SummaryWriter(log_dir=LOG_DIR)
    logger.info(f"TensorBoard Writer initialized at: {LOG_DIR}")

    # --- 读取完整训练数据 ---
    full_df = pd.read_csv(TRAIN_CSV_PATH)
    full_df = full_df.reset_index(drop=True)
    logger.info(f"完整训练数据集大小: {len(full_df)}")

    # 初始化编码器
    clinical_encoder = ClinicalEncoder(full_df, TEXT_COLS)
    clinical_dim = clinical_encoder.clinical_dim
    logger.info(f"临床特征维度: {clinical_dim}")

    # --- 五折划分 ---
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_idx = 1
    all_conf_matrices = []
    fold_final_metrics = []

    for train_idx, val_idx in kf.split(full_df):
        logger.info(f"\n===== 开始第 {fold_idx} 折训练 =====")
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

        # --- 创建 Dataset 和 DataLoader ---
        def create_dataset(df, is_train=False):
            return MultiTaskImageDatasetFromDataFrame(
                df=df,
                img_col=IMAGE_PATH_COLUMN,
                label_cols=LABEL_COLUMNS,
                size=TARGET_IMAGE_SIZE,
                logger=logger,
                clinical_encoder=clinical_encoder,
                is_training=is_train
            )

        train_dataset = create_dataset(train_df, is_train=True)
        val_dataset = create_dataset(val_df, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)

        # --- 类别权重 ---
        task_weights = {}
        num_classes_dict = {task: len(full_df[task].unique()) for task in LABEL_COLUMNS}
        for task in LABEL_COLUMNS:
            labels = train_df[task].values
            valid_labels = labels[labels != -1]
            if len(valid_labels) == 0:
                weight = torch.ones(num_classes_dict[task], device=DEVICE)
            else:
                classes = np.arange(num_classes_dict[task])
                weights = compute_class_weight('balanced', classes=classes, y=valid_labels)
                weight = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            task_weights[task] = weight

        # --- 初始化模型 ---
        logger.info(f"使用骨干网络: {MODEL_TYPE}")
        model = MultiTaskClassifier(num_classes_dict, clinical_dim, logger).to(DEVICE)

        criterion_dict = {}
        for task in LABEL_COLUMNS:
            num_cls = num_classes_dict[task]
            weight = task_weights[task]
            logger.info(f"[{task}] 类别权重:")
            for cls_idx, w in enumerate(weight):
                logger.info(f"  class {cls_idx}: {w.item():.4f}")
            if num_cls == 2:
                pos_weight_scalar = (weight[1] / weight[0]).detach().clone().to(DEVICE)
                criterion_dict[task] = nn.BCEWithLogitsLoss(pos_weight=pos_weight_scalar)
            else:
                criterion_dict[task] = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        scaler = torch.amp.GradScaler('cuda')

        best_val_score = -1.0
        best_model_path = os.path.join(LOG_DIR, f"best_model_fold{fold_idx}.pth")
        # ================= 训练 =================
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            train_labels_all = {task: [] for task in LABEL_COLUMNS}
            train_probs_all = {task: [] for task in LABEL_COLUMNS}
            for batch in train_loader:
                if batch is None:
                    continue

                pixel_values, clinical_values, labels_dict, _ = batch
                pixel_values = pixel_values.to(DEVICE)
                clinical_values = clinical_values.to(DEVICE)

                if torch.isnan(pixel_values).any():
                    print("image has nan")

                if torch.isnan(clinical_values).any():
                    print("clinical has nan")
                for task in labels_dict:
                    labels_dict[task] = labels_dict[task].to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    preds = model(pixel_values, clinical_values)
                    loss = 0
                    for task in LABEL_COLUMNS:
                        logits = preds[task]
                        labels = labels_dict[task]
                        task_criterion = criterion_dict[task]
                        valid_mask = labels != -1
                        if valid_mask.sum() == 0:
                            continue
                        valid_logits = logits[valid_mask]
                        valid_labels = labels[valid_mask]
                        if num_classes_dict[task] == 2:
                            target = valid_labels.float().view(-1, 1)
                            task_loss = task_criterion(valid_logits, target)
                            # 计算全部样本的概率 (用于指标统计)
                            probs_pos = torch.sigmoid(valid_logits).squeeze(1)
                            # 重新构造 [1-p, p] 格式的概率，用于 metrics
                            probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1) 
                        else:
                            target = valid_labels
                            probabilities = torch.softmax(valid_logits , dim=1)
                        loss += task_loss
                        train_labels_all[task].extend(valid_labels.cpu().tolist())
                        train_probs_all[task].extend(probabilities.cpu().tolist())
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            # ================= 验证 =================
            logger.info(f"--------- Epoch {epoch + 1} 训练评估总结 --------")
            train_metrics = {}
            for task_name in LABEL_COLUMNS:
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

            val_metrics, val_paths, val_labels, val_probs, val_pred_probs = evaluate(model,val_loader,criterion_dict,LABEL_COLUMNS,num_classes_dict,DEVICE,mode='val',logger=logger)
            val_score = np.mean([val_metrics[t]['auroc'] for t in LABEL_COLUMNS])
            logger.info(
                f"Fold {fold_idx} | Epoch {epoch+1} | "
                f"Loss {total_loss:.4f} | Val AUROC {val_score:.4f}"
            )

            # ================ 保存最佳模型 =================
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_val_score': val_score
                    },
                    best_model_path
                )
                logger.info(f"⭐ New Best Model Saved: {val_score:.4f}")

        # ================= 加载最佳模型 =================
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info(
                f"加载最佳模型 | Epoch {checkpoint['epoch']} | "
                f"Score {checkpoint['best_val_score']:.4f}"
            )
            fold_final_metrics.append({
                'fold': fold_idx,
                'avg_auroc': checkpoint['best_val_score'],
                # 如果你想记录每个任务的具体指标，可以从 val_metrics 中提取
                'task_detail': val_metrics 
            })
        fold_idx += 1
        # ================= 混淆矩阵 =================

        model.eval()
        all_labels = []
        all_preds = []
        task = LABEL_COLUMNS[0]
        with torch.no_grad():
            for batch in val_loader:
                pixel_values, clinical_values, labels_dict, _ = batch
                pixel_values = pixel_values.to(DEVICE)
                clinical_values = clinical_values.to(DEVICE)
                preds = model(pixel_values, clinical_values)
                logits = preds[task]
                labels = labels_dict[task]

                valid_mask = labels != -1

                if valid_mask.sum() == 0:
                    continue

                logits = logits[valid_mask]
                labels = labels[valid_mask].cpu().numpy()

                if num_classes_dict[task] == 2:

                    probs = torch.sigmoid(logits).squeeze(1)
                    preds_label = (probs > 0.5).long().cpu().numpy()

                else:

                    preds_label = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(labels)
                all_preds.extend(preds_label)
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"第 {fold_idx} 折混淆矩阵:\n{cm}")
        all_conf_matrices.append(cm)
        fold_idx += 1
    avg_cm = sum(all_conf_matrices)
    logger.info(f"\n===== 平均混淆矩阵 =====\n{avg_cm}")
    writer.close()
    return model

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
        trained_model.eval()
        if not USE_CLINICAL:
            main_logger.info("USE_CLINICAL=False，使用纯图像模式，无门控融合信息可分析。")
        else:
            for task_name in LABEL_COLUMNS:
                # g 的形状为 [Batch_Size, 512]
                g_tensor = trained_model.classifiers[task_name].last_g
                if g_tensor is None:
                    main_logger.info(f"任务名称: {task_name} — last_g 为空，跳过分析")
                    continue

                # 计算平均值
                avg_g = g_tensor.mean().item()

                main_logger.info(f"任务名称: {task_name}")
                main_logger.info(f"  - g 值矩阵形状: {list(g_tensor.shape)}")
                main_logger.info(f"  - 平均图像权重比例: {avg_g:.4f}")

                # 判断偏向
                bias = "图像 (Image)" if avg_g > 0.5 else "临床 (Clinical)"
                main_logger.info(f"  - 决策偏向结论: 更加依赖 {bias}")

                # 如果你想看每个样本的具体分数（512维取均值后）
                sample_gs = g_tensor.mean(dim=1).cpu().numpy()
                main_logger.info(f"  - 该批次前5个样本的 g 分数: {sample_gs[:5]}")
                main_logger.info("-" * 30)
                
