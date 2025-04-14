import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from cls import train_and_evaluate_model
from D_privacy import apply_differential_privacy

# 图像保存目录（src 同级）
output_dir = "../result_plots"

# 读取预处理数据
data_path = "../result/DataResult_A.csv"
data = pd.read_csv(data_path)
X = data.drop(" Label", axis=1)
y = data[" Label"]

# 删除 NaN 标签对应的行
X = X[y.notna()]
y = y[y.notna()]

# 标签分布图
plt.figure(figsize=(8, 4))
sns.countplot(x=y)
plt.title('Label Distribution in Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/label_distribution.png")
plt.close()

# Top-20 特征相关性热力图
corr = X.corr().iloc[:20, :20]
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Top 20 Feature Correlations')
plt.tight_layout()
plt.savefig(f"{output_dir}/top20_corr.png")
plt.close()

# 第一个特征的分布直方图 + KDE
plt.figure(figsize=(12, 6))
sns.histplot(data=X.iloc[:, 0], kde=True)
plt.title(f'Distribution of First Feature: {X.columns[0]}')
plt.tight_layout()
plt.savefig(f"{output_dir}/feature1_distribution.png")
plt.close()

# 全部特征的相关性热力图
plt.figure(figsize=(20, 16))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Full Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(f"{output_dir}/full_corr_matrix.png")
plt.close()

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="PortScan", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="PortScan", zero_division=0)
    return acc, prec, rec

def main(use_dp=False):
    if use_dp:
        print(f"[INFO] Applying Differential Privacy to training data... (ε=1.0)")
        X_train_dp = apply_differential_privacy(X_train, epsilon=1.0)
        y_train_used = y_train
        print("[INFO] Training model with DP-enabled data...")
        y_pred = train_and_evaluate_model(X_train_dp, y_train_used, X_test)
    else:
        print("[INFO] Training model without Differential Privacy...")
        y_pred = train_and_evaluate_model(X_train, y_train, X_test)

    acc, prec, rec = evaluate(y_test, y_pred)
    print(f"\n[Test Set Performance]\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    with open("system_log.txt", "a") as f:
        mode = "DP" if use_dp else "No-DP"
        f.write(f"[{mode}] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", action="store_true", help="Use Differential Privacy")
    args = parser.parse_args()
    main(use_dp=args.dp)