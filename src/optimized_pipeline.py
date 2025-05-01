# optimized_pipeline.py

import argparse
import logging
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from cls import train_and_evaluate_model


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def apply_feature_dp(X_norm: np.ndarray, epsilon: float) -> np.ndarray:
    """
    在归一化特征上注入 Laplace 噪声（敏感度=1）。
    """
    if epsilon <= 0:
        return X_norm
    scale = 1.0 / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=X_norm.shape)
    return X_norm + noise


def oversample_training(X: pd.DataFrame, y: pd.Series, labels: list):
    """
    对少数类进行上采样，使两类样本数持平。
    """
    df_train = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df_train.columns = list(X.columns) + ['Label']
    majority = df_train[df_train['Label'] == labels[0]]
    minority = df_train[df_train['Label'] != labels[0]]
    if minority.empty:
        return X, y
    minority_up = resample(minority,
                           replace=True,
                           n_samples=len(majority),
                           random_state=42)
    df_bal = pd.concat([majority, minority_up])
    X_bal = df_bal.drop('Label', axis=1)
    y_bal = df_bal['Label']
    return X_bal, y_bal


def clip_and_normalize(X: pd.DataFrame, fmin: pd.Series, fmax: pd.Series) -> np.ndarray:
    """
    将特征裁剪到 [fmin, fmax] 并线性归一化到 [0,1]。
    """
    Xc = X.clip(lower=fmin, upper=fmax, axis=1)
    return ((Xc - fmin) / (fmax - fmin)).values


def setup_logging(log_path: Path):
    """
    配置日志输出到文件和终端。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
    )


def load_config(config_file: Path) -> dict:
    """
    从 YAML 文件加载配置。
    """
    with open(config_file) as f:
        return yaml.safe_load(f)


def prepare_output_dirs(base_dir: Path):
    """
    创建 plots/ 和 logs/ 子目录。
    """
    plots_dir = base_dir / 'plots'
    logs_dir = base_dir / 'logs'
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, logs_dir


def load_and_split(data_path: Path, test_size: float, random_state: int, stratify: bool):
    """
    读取 CSV，拆分训练/测试集。
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    if 'Label' not in df.columns:
        raise KeyError(f"Label column not found. Available: {df.columns.tolist()}")
    df = df.dropna(subset=['Label'])
    X = df.drop('Label', axis=1)
    y = df['Label'].astype(str)
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def generate_epsilons(exp_cfg: dict):
    """
    根据 config 中的 epsilon_range 生成 eps 值列表。
    """
    er = exp_cfg.get('epsilon_range', {})
    min_eps = er.get('min', 0.0)
    max_eps = er.get('max', 2.0)
    step = er.get('step', 0.25)
    epsilons = np.arange(min_eps, max_eps + 1e-8, step).tolist()
    plot_scale = er.get('scale', 'linear')
    return epsilons, plot_scale


def evaluate_metrics(y_true, y_pred, labels):
    """
    计算 Accuracy, Precision, Recall (macro), F1 (macro) 及各类 recall/F1。
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_recalls = {
        f'recall_{labels[i]}': cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        for i in range(len(labels))
    }
    per_class_f1 = {}
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        per_class_f1[f'f1_{lbl}'] = (2 * tp / denom) if denom > 0 else 0

    return acc, prec, rec_macro, f1_macro, per_class_recalls, per_class_f1


def plot_confusion_matrix_grid(cms, labels, title, save_path: Path):
    """
    将所有 ε 下的混淆矩阵绘制为网格图。
    """
    n = len(cms)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    for ax, (eps, cm) in zip(axes, cms):
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_title(f'ε={eps:.2f}', fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=6)

    for ax in axes[n:]:
        ax.axis('off')

    fig.subplots_adjust(right=0.85, wspace=0.4, hspace=0.6)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(title, y=0.98)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_combined_curves(roc_data, pr_data, save_path: Path, scale: str):
    """
    在同一张图中并排展示所有 ε 的 ROC & PR 曲线。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ROC
    for eps, fpr, tpr, roc_auc_val in roc_data:
        ax1.plot(fpr, tpr, label=f'ε={eps:.2f} (AUC={roc_auc_val:.2f})')
    ax1.plot([0, 1], [0, 1], '--', color='grey')
    ax1.set_title('ROC Curves')
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    if scale == 'log':
        ax1.set_xscale('log')
    ax1.legend(fontsize=6, bbox_to_anchor=(1, 1))
    ax1.grid(linestyle='--', alpha=0.5)

    # PR
    for eps, rec_vals, prec_vals, pr_auc_val in pr_data:
        ax2.plot(rec_vals, prec_vals, label=f'ε={eps:.2f} (AUC={pr_auc_val:.2f})')
    ax2.set_title('PR Curves')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    if scale == 'log':
        ax2.set_xscale('log')
    ax2.legend(fontsize=6, bbox_to_anchor=(1, 1))
    ax2.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_results_table(results_df: pd.DataFrame, save_path: Path):
    """
    将结果 DataFrame 渲染为表格图片。
    """
    fig, ax = plt.subplots(figsize=(results_df.shape[1]*1.2, results_df.shape[0]*0.6))
    ax.axis('off')
    table = ax.table(
        cellText=np.round(results_df.values, 4),
        colLabels=results_df.columns,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

def run_experiments(config: dict):
    # 1) 配置 & 日志
    data_cfg = config['data']
    exp_cfg = config['experiment']
    output_cfg = config['output']
    plots_dir, logs_dir = prepare_output_dirs(Path(output_cfg['base_dir']))
    setup_logging(logs_dir / 'pipeline.log')

    # 2) 加载 & 拆分数据
    X_train, X_test, y_train, y_test = load_and_split(
        Path(data_cfg['path']),
        data_cfg.get('test_size', 0.3),
        data_cfg.get('random_state', 42),
        data_cfg.get('stratify', True)
    )

    # 3) 特征裁剪 & 归一化
    fmin, fmax = X_train.min(), X_train.max()
    X_train_norm = clip_and_normalize(X_train, fmin, fmax)
    X_test_norm = clip_and_normalize(X_test, fmin, fmax)

    # 4) ε 列表 & 容器
    epsilons, plot_scale = generate_epsilons(exp_cfg)
    metrics_list, cms, roc_data, pr_data = [], [], [], []
    labels = exp_cfg.get('labels', ['BENIGN', 'PortScan'])
    oversample = exp_cfg.get('oversample', False)

    # 5) 遍历 ε
    for eps in epsilons:
        logging.info(f'Running experiment ε={eps:.2f}')

        X_tr, y_tr = X_train_norm.copy(), y_train
        X_tr = apply_feature_dp(X_tr, eps)

        if oversample:
            Xdf = pd.DataFrame((X_tr * (fmax - fmin) + fmin), columns=X_train.columns)
            Xdf, y_tr = oversample_training(Xdf, y_tr, labels)
            X_tr = clip_and_normalize(Xdf, fmin, fmax)

        y_pred, y_prob = train_and_evaluate_model(
            X_tr, y_tr, X_test_norm, return_proba=True
        )

        cms.append((eps, confusion_matrix(y_test, y_pred, labels=labels)))

        acc, prec, rec_macro, f1_macro, rec_pc, f1_pc = evaluate_metrics(
            y_test, y_pred, labels
        )
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label='PortScan')
        roc_auc_val = auc(fpr, tpr)
        p_vals, r_vals, _ = precision_recall_curve(
            y_test, y_prob[:, 1], pos_label='PortScan'
        )
        pr_auc_val = auc(r_vals, p_vals)

        roc_data.append((eps, fpr, tpr, roc_auc_val))
        pr_data.append((eps, r_vals, p_vals, pr_auc_val))

        metrics_list.append({
            'epsilon': eps,
            'accuracy': acc,
            'precision': prec,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'roc_auc': roc_auc_val,
            'pr_auc': pr_auc_val,
            **rec_pc, **f1_pc
        })

    # 6) 保存 & 绘图
    plot_confusion_matrix_grid(
        cms, labels, 'Confusion Matrices for All ε',
        plots_dir / 'confusion_matrix_grid.png'
    )

    results_df = pd.DataFrame(metrics_list)
    results_df.to_csv(
        Path(output_cfg['base_dir']) / output_cfg.get('results_file', 'results_summary.csv'),
        index=False
    )
    plot_results_table(results_df, plots_dir / 'results_table.png')
    plot_combined_curves(
        roc_data, pr_data,
        plots_dir / 'combined_roc_pr.png',
        plot_scale
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DP anomaly detection pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    run_experiments(cfg)