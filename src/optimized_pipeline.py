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
from D_privacy import apply_differential_privacy


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
    )


def load_config(config_file: Path) -> dict:
    with open(config_file) as f:
        return yaml.safe_load(f)


def prepare_output_dirs(base_dir: Path):
    plots_dir = base_dir / 'plots'
    logs_dir = base_dir / 'logs'
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, logs_dir


def load_and_split(data_path: Path, test_size: float, random_state: int, stratify: bool):
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
    er = exp_cfg.get('epsilon_range', {})
    min_eps = er.get('min', 0.0)
    max_eps = er.get('max', 2.0)
    step = er.get('step', 0.25)
    epsilons = np.arange(min_eps, max_eps + 1e-8, step).tolist()
    plot_scale = er.get('scale', 'linear')
    return epsilons, plot_scale


def oversample_training(X: pd.DataFrame, y: pd.Series, labels: list):
    df_train = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df_train.columns = list(X.columns) + ['Label']
    majority = df_train[df_train['Label'] == labels[0]]
    minority = df_train[df_train['Label'] != labels[0]]
    if len(minority) == 0:
        return X, y
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    df_bal = pd.concat([majority, minority_upsampled])
    X_bal = df_bal.drop('Label', axis=1)
    y_bal = df_bal['Label']
    return X_bal, y_bal


def evaluate_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recalls = {f'recall_{labels[i]}': cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                         for i in range(len(labels))}
    per_class_f1 = {}
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        per_class_f1[f'f1_{lbl}'] = (2 * tp / denom) if denom > 0 else 0
    return acc, prec, rec_macro, f1_macro, per_class_recalls, per_class_f1


def plot_confusion_matrix_grid(cms, labels, title, save_path: Path):
    n = len(cms)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    for ax, (eps, cm) in zip(axes, cms):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
    # roc_data: list of (eps, fpr, tpr)
    # pr_data: list of (eps, recall, precision)
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
    for eps, recall_vals, precision_vals, pr_auc_val in pr_data:
        ax2.plot(recall_vals, precision_vals, label=f'ε={eps:.2f} (AUC={pr_auc_val:.2f})')
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
    fig, ax = plt.subplots(figsize=(results_df.shape[1]*1.2, results_df.shape[0]*0.6))
    ax.axis('off')
    table = ax.table(cellText=np.round(results_df.values, 4), colLabels=results_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_experiments(config: dict):
    data_cfg = config['data']
    exp_cfg = config['experiment']
    output_cfg = config['output']

    base_dir = Path(output_cfg['base_dir'])
    plots_dir, logs_dir = prepare_output_dirs(base_dir)
    setup_logging(logs_dir / 'pipeline.log')

    X_train, X_test, y_train, y_test = load_and_split(
        Path(data_cfg['path']), data_cfg.get('test_size', 0.3),
        data_cfg.get('random_state', 42), data_cfg.get('stratify', True)
    )

    epsilons, plot_scale = generate_epsilons(exp_cfg)
    metrics_list = []
    cms = []
    roc_data = []
    pr_data = []
    labels = exp_cfg.get('labels', ['BENIGN', 'PortScan'])
    oversample = exp_cfg.get('oversample', False)

    for eps in epsilons:
        logging.info(f'Running experiment ε={eps:.2f}')
        X_tr, y_tr = X_train, y_train
        if eps > 0:
            X_tr = apply_differential_privacy(X_tr, eps)
        if oversample:
            X_tr, y_tr = oversample_training(X_tr, y_tr, labels)

        y_pred, y_prob = train_and_evaluate_model(X_tr, y_tr, X_test, return_proba=True)
        cms.append((eps, confusion_matrix(y_test, y_pred, labels=labels)))

        acc, prec, rec_macro, f1_macro, rec_pc, f1_pc = evaluate_metrics(y_test, y_pred, labels)
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label='PortScan')
        roc_auc_val = auc(fpr, tpr)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob[:, 1], pos_label='PortScan')
        pr_auc_val = auc(recall_vals, precision_vals)

        # store for combined curves
        roc_data.append((eps, fpr, tpr, roc_auc_val))
        pr_data.append((eps, recall_vals, precision_vals, pr_auc_val))

        record = {
            'epsilon': eps,
            'accuracy': acc,
            'precision': prec,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'roc_auc': roc_auc_val,
            'pr_auc': pr_auc_val
        }
        record.update(rec_pc)
        record.update(f1_pc)
        metrics_list.append(record)

    # Plot confusion matrix grid
    grid_path = plots_dir / 'confusion_matrix_grid.png'
    plot_confusion_matrix_grid(cms, labels, 'Confusion Matrices for All ε', grid_path)

    # Save metrics table
    results_df = pd.DataFrame(metrics_list)
    csv_path = base_dir / output_cfg.get('results_file', 'results_summary.csv')
    results_df.to_csv(csv_path, index=False)
    table_path = plots_dir / 'results_table.png'
    plot_results_table(results_df, table_path)

    # Combined curves
    combined_curves_path = plots_dir / 'combined_roc_pr.png'
    plot_combined_curves(roc_data, pr_data, combined_curves_path, plot_scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DP-enabled anomaly detection pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(Path(args.config))
    run_experiments(config)