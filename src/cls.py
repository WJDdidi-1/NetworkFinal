
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_and_evaluate_model(X_train, y_train, X_test):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_labels = ["PortScan" if p == 1 else "BENIGN" for p in y_pred]

    print("\n[Train Set Classification Report]")
    print(classification_report(y_train, pipeline.predict(X_train), target_names=["BENIGN", "PortScan"]))

    cm = confusion_matrix(y_train, pipeline.predict(X_train))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["BENIGN", "PortScan"], yticklabels=["BENIGN", "PortScan"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Train)")
    plt.tight_layout()
    plt.savefig("../result_plots/confusion_matrix_train.png")
    plt.close()

    try:
        model = pipeline.named_steps['classifier']
        importance = abs(model.coef_[0])
        feature_names = X_train.columns
        top_idx = importance.argsort()[::-1][:15]
        top_feats = [feature_names[i] for i in top_idx]
        top_vals = [importance[i] for i in top_idx]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_vals, y=top_feats)
        plt.xlabel("Importance")
        plt.title("Top 15 Important Features")
        plt.tight_layout()
        plt.savefig("../result_plots/feature_importance.png")
        plt.close()
    except Exception as e:
        print("无法生成特征重要性图：", str(e))

    return y_pred_labels
