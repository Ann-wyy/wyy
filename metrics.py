import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, AutoModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
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
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T


# --- 评估函数 ---
def calculate_metrics(
    all_labels: List[int],
    all_probs: List[List[float]],  # shape: [N, C]
    num_classes: int,
    task_name: str,
    mode: str,
    logger: logging.Logger
)  -> Dict[str, float]:
    """计算二分类或多分类模型的评估指标。"""
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    metrics = {}
    valid_indices = (all_labels >= 0)
    
    # 过滤数据
    valid_labels = all_labels[valid_indices]
    valid_preds = all_preds[valid_indices]
    valid_probs = all_probs[valid_indices]
    if len(valid_labels) == 0:
        logger.warning(f"任务 {task_name} 中没有有效标签 (>=0) 的样本，跳过指标计算。")
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auroc': 0.0, 'auprc': 0.0}
    metrics['accuracy'] = float(accuracy_score(valid_labels, valid_preds))
    target_names = list(range(num_classes))
    metrics['precision'] = float(precision_score(valid_labels, valid_preds, labels=target_names,average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(valid_labels, valid_preds, labels=target_names,average='macro', zero_division=0))
    metrics['f1'] = float(f1_score(valid_labels, valid_preds, labels=target_names,average='macro', zero_division=0))

    logger.info(f"--- {mode.upper()} 结果 (类别数: {num_classes}) ---")
    logger.info(f"整体准确率 (Accuracy): {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"整体精确率 (Precision): {metrics['precision'] * 100:.2f}%")
    logger.info(f"整体召回率 (Recall): {metrics['recall'] * 100:.2f}%")
    logger.info(f"整体 F1-Score: {metrics['f1'] * 100:.2f}%")

    # --- AUROC ---
    try:
        if num_classes == 2:
            if len(np.unique(valid_labels)) < 2:
                raise ValueError("有效二分类样本只含有一个类别。")
            # 二分类：用正类概率
            auroc = roc_auc_score(valid_labels, valid_probs[:, 1]) # 使用过滤后的数据
            auprc = average_precision_score(valid_labels, valid_probs[:, 1])
        else:
            classes = list(range(num_classes))
            y_true_bin = label_binarize(valid_labels, classes=classes)
            auroc = roc_auc_score(y_true_bin, valid_probs, multi_class='ovr')
            auprc = average_precision_score(y_true_bin, valid_probs, average='macro')
        metrics['auprc'] = float(auprc)
        metrics['auroc'] = float(auroc)
        logger.info(f"整体 AUPRC: {metrics['auprc'] * 100:.2f}%")
        logger.info(f"整体 AUROC: {metrics['auroc'] * 100:.2f}%")
    except Exception as e:
        logger.warning(f"计算 AUROC/AUPRC 失败 ({task_name}, {mode}): {e}")
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0

    return metrics

        
# --- 辅助函数：评估流程 ---
def evaluate(
    model,
    data_loader,
    criterion_dict,
    task_names,
    num_classes_dict,
    device,
    mode,
    logger
):
    model.eval()

    task_labels = {task: [] for task in task_names}
    task_probs = {task: [] for task in task_names}

    total_combined_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue

            pixel_values, labels_dict, img_paths = batch
            if pixel_values is None or pixel_values.numel() == 0:
                continue

            pixel_values = pixel_values.to(device)

            predictions_dict = model(pixel_values)
            combined_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            for task_name in task_names:
                labels = labels_dict[task_name].to(device)
                predictions = predictions_dict[task_name]
                num_cls = int(num_classes_dict.get(task_name, 2))
                task_criterion = criterion_dict[task_name]

                # --- 统一 shape ---
                # labels: [B] / [B,1] -> [B]
                labels = labels.view(-1)

                # predictions:
                # multi-class -> [B,C]
                # binary -> [B] or [B,1] -> [B]
                if num_cls <= 2:
                    predictions = predictions.view(-1)

                # --- loss: 只对 labels != -1 的样本计算 ---
                valid_mask = (labels != -1)
                valid_count = int(valid_mask.sum().item())

                task_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                if valid_count > 0:
                    if num_cls > 2:
                        # 多分类
                        valid_labels = labels[valid_mask].long()  # [N]
                        valid_predictions = predictions[valid_mask]  # [N,C]
                        task_loss = task_criterion(valid_predictions, valid_labels)
                    else:
                        # 二分类 BCEWithLogitsLoss
                        valid_labels = labels[valid_mask].float()  # [N]
                        valid_predictions = predictions[valid_mask].float()  # [N]
                        task_loss = task_criterion(valid_predictions, valid_labels)

                combined_loss = combined_loss + task_loss

                # --- probabilities: 用全 batch 的 predictions（后续会按 labels 过滤） ---
                if num_cls > 2:
                    probabilities = torch.softmax(predictions, dim=1)  # [B,C]
                else:
                    probs_pos = torch.sigmoid(predictions)  # [B]
                    probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1)  # [B,2]

                task_probs[task_name].extend(probabilities.detach().cpu().tolist())
                task_labels[task_name].extend(labels.detach().cpu().tolist())

            total_combined_loss += float(combined_loss.item())
            total_steps += 1

    # ===== 计算指标 =====
    task_metrics = {}
    for task_name in task_names:
        true_labels = task_labels[task_name]
        probabilities = task_probs[task_name]

        if len(true_labels) == 0:
            task_metrics[task_name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auroc': 0.0,
                'auprc': 0.0
            }
            continue

        num_cls = int(num_classes_dict.get(task_name, 2))
        metrics = calculate_metrics(
            all_labels=true_labels,
            all_probs=probabilities,
            num_classes=num_cls,
            task_name=task_name,
            mode=f'{mode}_{task_name}',
            logger=logger
        )
        task_metrics[task_name] = metrics

    avg_loss = total_combined_loss / max(total_steps, 1)
    logger.info(f"[{mode}] Combined Loss (avg per step): {avg_loss:.6f}")

    return task_metrics



# --- 辅助函数：TensorBoard 记录 ---
def log_metrics_to_tensorboard(
    writer: SummaryWriter, 
    metrics_dict: Dict[str, Dict[str, float]], 
    step: int, 
    stage: str, 
    logger: logging.Logger,
):
    """
    将所有指标（accuracy, precision, recall, f1, auroc, auprc）按任务类型聚合后写入 TensorBoard。
    """
    all_task_names = list(metrics_dict.keys()) 
    independent_tasks = [task for task in all_task_names]

    # 所有独立子任务（用于总体平均）
    all_individual_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auroc': [],
        'auprc': []
    }

    logger.info(f"--- {stage} Epoch {step} 任务摘要指标 ---")

    # 2. 处理独立任务
    for task_name in independent_tasks:
        metrics = metrics_dict.get(task_name, {})
        if not metrics:
            continue
        for key in all_individual_metrics:
            val = metrics.get(key, float('nan'))
            if not np.isnan(val):
                writer.add_scalar(f'{stage}_Summary/{key.upper()}_{task_name}', val, step)
                all_individual_metrics[key].append(val)
    
    average_metrics = {}
    for key in [ 'auprc','auroc', 'accuracy']:
        values = all_individual_metrics[key]
        if values:
            avg_val = np.mean(values)
            average_metrics[key] = avg_val
            writer.add_scalar(f'{stage}_Aggregated/AVERAGE_{key.upper()}', avg_val, step)
            logger.info(f"Average {key.upper()}: {avg_val:.4f}")
        else:
            logger.warning(f"No valid {key.upper()} values found for averaging.")
            average_metrics[key] = float('nan')
    


def run_test_and_save_predictions(
    model,
    test_loader,
    task_name: str,
    num_classes_dict,
    device,
    test_df: pd.DataFrame,
    save_dir: str,
    logger: logging.Logger,
    is_save: bool = False,
    img_col: str = "image_path",
):
    """
    1) 对 test_loader 跑推理
    2) 保存逐图像预测结果 (image-level)
    3) 再按 patient_id 聚合为病人级结果 (patient-level)
    4) 计算病人级指标

    ✅ 支持多任务：所有输出列名都带 task_name 前缀，避免覆盖
    """

    model.eval()

    # ============ 检查任务类别数 ============
    num_cls = int(num_classes_dict.get(task_name, 2))
    if num_cls != 2:
        raise ValueError(
            f"run_test_and_save_predictions 当前只支持二分类，但 {task_name} num_classes={num_cls}"
        )

    # ============ image-level 收集 ============
    all_image_paths = []
    all_gt = []
    all_prob = []
    all_pred = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            pixel_values, labels_dict, img_paths = batch

            if pixel_values is None or pixel_values.numel() == 0:
                continue
            if img_paths is None or len(img_paths) == 0:
                continue

            pixel_values = pixel_values.to(device)

            if task_name not in labels_dict:
                raise KeyError(
                    f"❌ labels_dict 缺少任务 {task_name}，现有 keys={list(labels_dict.keys())}"
                )

            labels = labels_dict[task_name].to(device).view(-1)  # [B]

            logits_dict = model(pixel_values)

            if task_name not in logits_dict:
                raise KeyError(
                    f"❌ logits_dict 缺少任务 {task_name}，现有 keys={list(logits_dict.keys())}"
                )

            logits = logits_dict[task_name].view(-1)  # [B]

            probs_pos = torch.sigmoid(logits)  # [B]
            preds = (probs_pos >= 0.5).long()  # [B]

            all_image_paths.extend(list(img_paths))
            all_gt.extend(labels.detach().cpu().tolist())
            all_prob.extend(probs_pos.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    # ============ image-level dataframe (列名带任务前缀) ============
    gt_col = f"{task_name}_gt"
    pred_col = f"{task_name}_pred"
    prob_col = f"{task_name}_prob"

    pred_df = pd.DataFrame({
        "image_path": all_image_paths,
        gt_col: all_gt,
        prob_col: all_prob,
        pred_col: all_pred,
    })

    # ============ merge patient_id ============
    if "patient_id" not in test_df.columns:
        raise ValueError("❌ test_df 缺少 patient_id 列，请确认预处理是否保留了 patient_id")

    if img_col not in test_df.columns:
        raise ValueError("❌ test_df 缺少 image_path 列，无法 merge")

    meta_df = test_df[[img_col, "patient_id"]].copy()
    meta_df = meta_df.drop_duplicates(subset=[img_col], keep="first")

    pred_df = pred_df.merge(meta_df, on=img_col, how="left")

    missing_pid = int(pred_df["patient_id"].isna().sum())
    if missing_pid > 0:
        logger.error(f"❌ merge 后有 {missing_pid} 张图没有匹配到 patient_id（image_path 不一致）")
        logger.error("示例（前10个）：")
        logger.error(pred_df[pred_df["patient_id"].isna()][img_col].head(10).tolist())
        raise ValueError("merge 失败：image_path 与 test_df 不一致，导致 patient_id 缺失")

    # ============ 保存逐图像结果 ============
    if is_save:
        os.makedirs(save_dir, exist_ok=True)
        image_csv_path = os.path.join(save_dir, f"test_image_level_{task_name}.csv")
        pred_df.to_csv(image_csv_path, index=False)
        logger.info(f"✅ 已保存逐图像预测结果: {image_csv_path}")

    # ============ patient-level 聚合 (列名带任务前缀) ============
    patient_gt_col = f"{task_name}_patient_gt"
    patient_pred_col = f"{task_name}_patient_pred"
    patient_prob_col = f"{task_name}_patient_prob"
    patient_nimg_col = f"{task_name}_n_images"

    patient_df = pred_df.groupby("patient_id").agg(
        **{
            patient_gt_col: (gt_col, "max"),
            patient_pred_col: (pred_col, "max"),
            patient_prob_col: (prob_col, "max"),
            patient_nimg_col: ("image_path", "count"),
        }
    ).reset_index()

    if is_save:
        patient_csv_path = os.path.join(save_dir, f"test_patient_level_{task_name}.csv")
        patient_df.to_csv(patient_csv_path, index=False)
        logger.info(f"✅ 已保存病人级预测结果: {patient_csv_path}")

    # ============ patient-level 指标 ============
    logger.info(f"patient_df columns = {patient_df.columns.tolist()}")

    y_true = patient_df[patient_gt_col].to_numpy()
    y_pred = patient_df[patient_pred_col].to_numpy()
    y_prob = patient_df[patient_prob_col].to_numpy()

    # 过滤掉 gt = -1
    valid_mask = (y_true != -1)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    y_prob = y_prob[valid_mask]

    metrics = {}

    if len(y_true) == 0:
        logger.warning("❗ patient-level: 没有有效标签（全是 -1），无法计算指标")
        metrics = {
            f"{task_name}_accuracy": 0.0,
            f"{task_name}_precision": 0.0,
            f"{task_name}_recall": 0.0,
            f"{task_name}_f1": 0.0,
            f"{task_name}_auroc": float("nan"),
            f"{task_name}_auprc": float("nan"),
        }
        return metrics, pred_df, patient_df

    metrics[f"{task_name}_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics[f"{task_name}_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics[f"{task_name}_recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics[f"{task_name}_f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        metrics[f"{task_name}_auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics[f"{task_name}_auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        logger.warning(
            f"❗ patient-level: 只有单类 {unique_classes.tolist()}，AUROC/AUPRC 无法计算，设为 NaN"
        )
        metrics[f"{task_name}_auroc"] = float("nan")
        metrics[f"{task_name}_auprc"] = float("nan")

    logger.info(f"========== 病人级 (OR聚合) Test Metrics [{task_name}] ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.6f}")
        else:
            logger.info(f"{k}: {v}")

    return metrics, pred_df, patient_df
