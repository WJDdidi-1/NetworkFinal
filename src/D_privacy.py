# D_privacy.py

import numpy as np
import pandas as pd

def apply_differential_privacy(df, epsilon=1.0, delta=1e-5, method='laplace'):
    """
    在输入 DataFrame 上应用差分隐私机制（Laplace 或 Gaussian）

    参数：
        df: pd.DataFrame（仅特征数据）
        epsilon: 隐私预算
        delta: 用于 Gaussian 机制的隐私参数
        method: 'laplace' 或 'gaussian'

    返回：
        带噪声的 pd.DataFrame
    """

    df_numeric = df.select_dtypes(include=[np.number])
    sensitivity = 1.0  # 假设已经归一化后的数据

    if method == 'laplace':
        scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=df_numeric.shape)
    elif method == 'gaussian':
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(loc=0, scale=sigma, size=df_numeric.shape)
    else:
        raise ValueError("method must be 'laplace' or 'gaussian'")

    df_noisy = df_numeric + noise
    df_noisy.columns = df_numeric.columns
    return df_noisy