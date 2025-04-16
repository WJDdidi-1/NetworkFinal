import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from cls import train_and_evaluate_model
from D_privacy import apply_differential_privacy

# 图像输出目录
output_dir = "../result_plots"
os.makedirs(output_dir, exist_ok=True)

# 数据加载
data_path = "../result/DataResult_A.csv"
data = pd.read_csv(data_path)
X = data.drop(" Label", axis=1)
y = data[" Label"].astype(str)  # 强制转为字符串，确保一致性

# 删除 NaN 标签
X = X[y.notna()]
y = y[y.notna()]

# 数据划分（推荐加 stratify，保持标签分布）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 评估 + 混淆矩阵绘图函数
def evaluate_and_plot(y_true, y_pred, label, filename_suffix):
    y_true = y_true.astype(str)
    y_pred = pd.Series(y_pred).astype(str)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=["BENIGN", "PortScan"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["BENIGN", "PortScan"],
                yticklabels=["BENIGN", "PortScan"])
    plt.title(f'Confusion Matrix ({label})')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_{filename_suffix}.png")
    plt.close()

    return acc, prec, rec

# 主运行逻辑
def run_single(epsilon=None):
    if epsilon is not None:
        print(f"[INFO] Applying Differential Privacy (ε={epsilon})")
        X_train_dp = apply_differential_privacy(X_train, epsilon=epsilon)
        y_pred = train_and_evaluate_model(X_train_dp, y_train, X_test)
        label = f"DP (ε={epsilon})"
        suffix = f"dp_eps{str(epsilon).replace('.', '')}"
    else:
        print("[INFO] Training without Differential Privacy")
        y_pred = train_and_evaluate_model(X_train, y_train, X_test)
        label = "No DP"
        suffix = "nodp"

    acc, prec, rec = evaluate_and_plot(y_test, y_pred, label, suffix)
    print(f"\n[{label} Results] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    with open("system_log.txt", "a") as f:
        f.write(f"[{label}] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}\n")

# 命令行主入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy mode")
    parser.add_argument("--epsilons", nargs='*', type=float, help="Set multiple epsilon values for DP")
    args = parser.parse_args()

    if args.dp:
        if args.epsilons:
            for eps in args.epsilons:
                run_single(epsilon=eps)
        else:
            run_single(epsilon=1.0)
    else:
        run_single()