# cls.py
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_model(X_train, y_train, X_test, return_proba=False):
    """
    训练模型并返回预测结果。
    参数:
      - return_proba: 如果为 True，则返回 (y_pred, y_prob)，否则仅返回 y_pred。
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if return_proba:
        y_prob = model.predict_proba(X_test)
        return y_pred, y_prob
    return y_pred