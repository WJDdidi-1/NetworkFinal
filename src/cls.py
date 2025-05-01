# cls.py

from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_model(X_train, y_train, X_test, return_proba=False):
    """
    训练一个随机森林模型并返回预测结果。
    如果 return_proba=True，则返回 (y_pred, y_prob)，
    否则只返回 y_pred。
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if return_proba:
        y_prob = model.predict_proba(X_test)
        return y_pred, y_prob

    return y_pred