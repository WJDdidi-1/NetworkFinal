# NetworkFinal — 隐私保护下的端口扫描检测

课程项目：在 **CICIDS 风格的网络流特征**上识别 `PortScan` 与 `BENIGN`，并回答一个
工程问题——**给特征加差分隐私噪声之后，检测能力掉多少？**

`src/main.py` 扫一遍 ε 列表（`null` 为不加噪声的基线），每个 ε 训练一次分类器、
出一次指标与混淆矩阵，最后把「ε → 准确率/精确率/召回率」画成图。

## 流程

```
result/DataResult_A.csv          流特征 + Label（BENIGN / PortScan）
      │
      ├─ src/cls.py              训练 + 评估（sklearn）
      ├─ src/D_privacy.py        Laplace / Gaussian 噪声，σ 由 ε 与 δ 决定
      └─ src/main.py             ε 扫描 → 指标 + 混淆矩阵 + 曲线
                                 → result_plots/ 、result_output/
```

`src/config.yaml` 固定了数据路径、`test_size: 0.3`、`random_state: 42`、
`stratify: true`，以及要扫的 ε 列表（`null / 0.5 / 1.0 / …`）。`src/optimized_pipeline.py`
是后续整理过的一版流程，`src/Network_A.ipynb`、`src/cls.ipynb` 是探索记录。

## 怎么跑

```console
$ pip install -r requirements.txt
$ cd src
$ python main.py --config config.yaml
```

输出落在 `result_plots/`（混淆矩阵、ε 曲线）与 `result_output/`（指标表）。
`Report/数据分析报告（简略版）.docx` 是课程报告，`result/` 里是切好的数据。

## 差分隐私这一步做了什么

`apply_differential_privacy(df, epsilon, delta, method)` 只对**数值列**加噪：

- Laplace：`scale = sensitivity / ε`，`sensitivity` 取 1.0（假设特征已归一化）。
- Gaussian：`σ = sqrt(2·ln(1.25/δ)) · sensitivity / ε`。
- 非数值列（`Label`）不进模型输入，因此不返回。

六条测试（`tests/test_privacy.py`）钉住了这部分行为：只返回数值列、形状不变、
噪声确实改变数值、ε 越大扰动越小、Gaussian 也加噪、未知机制抛 `ValueError`。

```console
$ python -m pytest -q        # 6 passed
```

## 限制

- `sensitivity = 1.0` 是写死的假设，不是从数据算出来的；特征未归一化时这个假设不成立。
- 逐特征独立加噪并不等于对「整条记录」的 ε-差分隐私，报告里应说明这一点。
- 数据集（`result/DataResult_A.csv`）是课程给的一份切片，不能当作 CICIDS 全量结论。
- 分类器与随机种子固定，单次运行的指标波动不大，但没有做交叉验证。

## 许可

没有 LICENSE 文件，按默认版权处理。
