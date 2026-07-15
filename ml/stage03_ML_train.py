"""
stage03_ML_train.py — 多标签疾病预测训练流水线
================================================
针对 data_stage02.csv（19 标签极端稀疏矩阵）设计。

核心问题：
    - 19 个标签中仅 4.7% 是非零元素（95.3% 稀疏）
    - 最稀有病（乙肝）仅 0.39% 正样本，最密（蛀牙）15.64%
    - 传统 accuracy / Top-k 在这种场景下评估分数虚高但无意义

解决方案：
    1. 模型：ClassifierChain + XGBoost（利用标签间相关性 + 梯度提升树）
    2. 类别权重：per-label scale_pos_weight 自动平衡
    3. 评估：F1-macro / F1-micro / AUC-PR / Hamming Loss / Subset Accuracy
    4. 阈值优化：逐标签 PR 曲线寻找最优决策阈值（替代默认 0.5）
    5. 分层 K-折交叉验证：保持标签分布

依赖：
    pip install xgboost scikit-learn matplotlib
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体回退
_plt_font_rc = {
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(_plt_font_rc)
import xgboost as xgb
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# 抑制 XGBoost 兼容性警告
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / "data"
_RESULT_DIR = _SCRIPT_DIR.parent / "data" / "intermediate"
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = _DATA_DIR / "data_stage02.csv"
MODEL_DIR = _SCRIPT_DIR.parent / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据加载与划分
# ============================================================


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """加载 stage02 预处理后的数据。"""
    logger.info(f"[加载] 读取 {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    logger.info(f"[加载] shape={df.shape}")
    return df


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 DataFrame 中切出 X 和 y。

    约定（由 stage02 reorder_columns 保证）：
        - 标签列排在最后，以 LABEL_PREFIX 开头
        - 其余列均为特征
    """
    from ml.stage02_preprocess import LABEL_PREFIX

    label_cols = [c for c in df.columns if c.startswith(LABEL_PREFIX)]
    feature_cols = [c for c in df.columns if not c.startswith(LABEL_PREFIX)]
    X = df[feature_cols].copy()
    y = df[label_cols].copy()
    logger.info(f"[切分] X={X.shape}, y={y.shape} ({len(label_cols)} 标签)")
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分层训练/测试集划分。

    策略：用每行患病标签数量作为分层依据，保证训练/测试集
    中"健康 vs 单病 vs 多病"的比例一致。
    """
    n_ill = y.sum(axis=1)
    # 将"患病数量"作为分层依据（0=健康, 1=单病, 2+=多病）
    strata = n_ill.clip(upper=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )
    logger.info(
        f"[划分] 训练 {len(X_train)} / 测试 {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# 模型构建
# ============================================================


def build_model(
    n_labels: int,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    scale_pos_weights: Optional[List[float]] = None,
    random_state: int = 42,
) -> ClassifierChain:
    """
    构建 ClassifierChain + XGBoost 多标签分类器。

    设计理由：
        - ClassifierChain：将标签 y_j 作为下一标签 y_{j+1} 的额外特征输入，
          利用标签间相关性（如"高血压"和"糖尿病"常共病）。
        - XGBoost：梯度提升树，天然支持不平衡数据（scale_pos_weight），
          对极端稀疏标签的表现远优于线性模型。

    参数：
        n_labels:         标签数量
        max_depth:        树深度（默认 6，兼顾拟合与泛化）
        learning_rate:    学习率（默认 0.1）
        n_estimators:     树的数量（默认 200）
        scale_pos_weights: 每个标签的正样本权重（列表，长度=n_labels）
                           自动计算：n_neg / n_pos
        random_state:     随机种子

    返回：
        拟合好的 ClassifierChain
    """
    if scale_pos_weights is None:
        scale_pos_weights = [1.0] * n_labels

    # 为每个标签构建 XGBoost 基分类器
    base_estimators = []
    for i in range(n_labels):
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weights[i],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state + i,
            n_jobs=4,
            verbosity=0,
        )
        base_estimators.append(clf)

    chain = ClassifierChain(
        base_estimators[0],
        order="random",
        random_state=random_state,
    )
    # ClassifierChain 只接受单个基分类器，内部 clone
    # 为了支持 19 个不同权重的分类器，换用 MultiOutputClassifier + 自定义链
    # 这里用更灵活的实现：
    from sklearn.multioutput import MultiOutputClassifier

    # 用 MultiOutputClassifier 包装所有基分类器
    # 再在外部手动实现链式传递
    chain._wrapped_estimator = MultiOutputClassifier(
        xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            scale_pos_weight=1.0,  # 每标签单独调
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=4,
            verbosity=0,
        ),
        n_jobs=4,
    )
    return chain


def compute_scale_pos_weights(y: pd.DataFrame) -> List[float]:
    """
    为每个标签计算 scale_pos_weight = n_neg / n_pos。

    这样稀有标签自动获得更高权重，缓解不平衡。
    """
    weights = []
    for col in y.columns:
        n_pos = int(y[col].sum())
        n_neg = len(y) - n_pos
        if n_pos == 0:
            w = 1.0
        else:
            w = n_neg / n_pos
        weights.append(round(float(w), 2))
    return weights


# ============================================================
# 自定义链式多标签分类器
# ============================================================


class XGBoostClassifierChain:
    """
    手动实现的 ClassifierChain + XGBoost。

    每个标签使用独立的 XGBoost 分类器，前 j-1 个标签的预测
    结果（概率）作为第 j 个标签的额外特征输入。

    相比 sklearn 的 ClassifierChain，本实现对每个标签支持
    独立的 scale_pos_weight。
    """

    def __init__(
        self,
        n_labels: int,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        scale_pos_weights: Optional[List[float]] = None,
        random_state: int = 42,
    ):
        self.n_labels = n_labels
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.scale_pos_weights = scale_pos_weights or [1.0] * n_labels
        self.random_state = random_state
        self.models: List[xgb.XGBClassifier] = []
        self.label_order_: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifierChain":
        n_samples, n_features = X.shape
        self.models = []
        self.label_order_ = list(range(self.n_labels))

        # 逐标签训练
        for j in range(self.n_labels):
            # 构建第 j 个标签的训练特征：原始特征 + 前 j 个标签的真值
            if j == 0:
                X_aug = X
            else:
                chain_features = y[:, :j].astype(np.float64)
                X_aug = np.hstack([X, chain_features])

            clf = xgb.XGBClassifier(
                objective="binary:logistic",
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                scale_pos_weight=self.scale_pos_weights[j],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state + j,
                n_jobs=4,
                verbosity=0,
            )
            clf.fit(X_aug, y[:, j])
            self.models.append(clf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 (n_samples, n_labels) 的概率矩阵。"""
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_labels), dtype=np.float64)

        for j in range(self.n_labels):
            if j == 0:
                X_aug = X
            else:
                chain_features = proba[:, :j]
                X_aug = np.hstack([X, chain_features])

            proba[:, j] = self.models[j].predict_proba(X_aug)[:, 1]

        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """返回 (n_samples, n_labels) 的二值预测矩阵。"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# ============================================================
# 阈值优化
# ============================================================


def find_best_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> np.ndarray:
    """
    为每个标签独立寻找最优决策阈值。

    方法：在验证集上遍历 [0.05, 0.95] 步长 0.05，
    选取使指定指标最大的阈值。

    参数：
        y_true:    (n_samples, n_labels) 真实标签
        y_proba:   (n_samples, n_labels) 预测概率
        metric:    优化目标 — "f1" / "precision" / "recall" / "fbeta"

    返回：
        thresholds: (n_labels,) 每个标签的最优阈值
    """
    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5)

    for j in range(n_labels):
        best_thresh = 0.5
        best_score = -1.0

        for t in np.arange(0.05, 0.96, 0.05):
            y_pred_j = (y_proba[:, j] >= t).astype(int)

            if metric == "f1":
                score = f1_score(y_true[:, j], y_pred_j, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true[:, j], y_pred_j, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true[:, j], y_pred_j, zero_division=0)
            else:
                score = f1_score(y_true[:, j], y_pred_j, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = t

        thresholds[j] = best_thresh

    return thresholds


# ============================================================
# 评估指标
# ============================================================


def evaluate_multilabel(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: List[str],
    thresholds: Optional[np.ndarray] = None,
) -> Dict:
    """
    全面评估多标签分类结果。

    指标：
        - Subset Accuracy：所有标签全对的样本比例（最严格）
        - Hamming Loss：平均每个标签的误判率
        - F1-macro：各标签 F1 等权平均（最常用）
        - F1-micro：全局 TP/FP/FN 计算
        - Precision-macro / Recall-macro
        - AUC-ROC（macro）：对不平衡数据较鲁棒
        - AUC-PR（macro）：比 AUC-ROC 更适合极端不平衡
        - Label Ranking Average Precision
        - 逐标签详细指标
    """
    n_labels = y_true.shape[1]
    if thresholds is None:
        thresholds = np.full(n_labels, 0.5)

    y_pred = (y_proba >= thresholds).astype(int)

    # 基础指标
    subset_acc = accuracy_score(y_true, y_pred)
    hl = hamming_loss(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC-ROC & AUC-PR
    try:
        auc_roc = roc_auc_score(y_true, y_proba, average="macro")
    except ValueError:
        auc_roc = 0.0
    try:
        auc_pr = average_precision_score(y_true, y_proba, average="macro")
    except ValueError:
        auc_pr = 0.0

    # Label Ranking AP
    try:
        lrap = label_ranking_average_precision_score(y_true, y_proba)
    except ValueError:
        lrap = 0.0

    # 逐标签指标
    per_label = {}
    for j, name in enumerate(label_names):
        per_label[name] = {
            "support": int(y_true[:, j].sum()),
            "f1": round(float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0)), 4),
            "precision": round(float(precision_score(y_true[:, j], y_pred[:, j], zero_division=0)), 4),
            "recall": round(float(recall_score(y_true[:, j], y_pred[:, j], zero_division=0)), 4),
            "threshold": round(float(thresholds[j]), 2),
        }

    result = {
        "subset_accuracy": round(float(subset_acc), 4),
        "hamming_loss": round(float(hl), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_micro": round(float(f1_micro), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "auc_roc_macro": round(float(auc_roc), 4),
        "auc_pr_macro": round(float(auc_pr), 4),
        "label_ranking_ap": round(float(lrap), 4),
        "per_label": per_label,
    }
    return result


# ============================================================
# 训练主流程
# ============================================================


def train_and_evaluate(
    data_path: Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    n_folds: int = 5,
    save_model: bool = True,
    plot: bool = True,
    norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """
    完整训练 + 评估流水线。

    流程：
        1. 加载数据
        2. 分层划分训练/测试集
        3. 计算 per-label scale_pos_weight
        4. 训练 XGBoostClassifierChain
        5. 在验证集上搜索最优阈值
        6. 在测试集上全面评估
        7. 保存模型和评估报告

    参数:
        norm_stats: stage02 输出的归一化统计量（每列 {"mean": float, "std": float}）。
                    None 时尝试从 data/intermediate/stage02_norm_stats.json 自动加载。

    返回评估结果字典。
    """
    from ml.stage02_preprocess import LABEL_PREFIX

    if norm_stats is None:
        norm_stats_path = _RESULT_DIR / "stage02_norm_stats.json"
        if norm_stats_path.exists():
            with open(norm_stats_path, "r", encoding="utf-8") as f:
                norm_stats = json.load(f)
            logger.info(f"[stage02] 已加载归一化统计量: {len(norm_stats)} 列")

    # ---------- 1. 加载 ----------
    df = load_data(data_path)
    X, y = split_X_y(df)
    label_names = list(y.columns)
    n_labels = len(label_names)

    # ---------- 2. 划分 ----------
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---------- 3. 计算类别权重 ----------
    scale_pos_weights = compute_scale_pos_weights(y_train)
    logger.info(f"[权重] scale_pos_weights: {dict(zip(label_names, scale_pos_weights))}")

    # ---------- 4. K-折交叉验证（选最佳超参） ----------
    logger.info(f"[CV] {n_folds} 折交叉验证...")
    cv_scores = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X_train, y_train.sum(axis=1).clip(upper=2)), 1
    ):
        X_tr, X_val = X_train.iloc[tr_idx].values, X_train.iloc[val_idx].values
        y_tr, y_val = y_train.iloc[tr_idx].values, y_train.iloc[val_idx].values

        chain = XGBoostClassifierChain(
            n_labels=n_labels,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=150,
            scale_pos_weights=scale_pos_weights,
            random_state=random_state,
        )
        chain.fit(X_tr, y_tr)
        proba_val = chain.predict_proba(X_val)

        # 快速 F1-macro 评估
        best_thresh = find_best_thresholds(y_val, proba_val, metric="f1")
        y_pred_val = (proba_val >= best_thresh).astype(int)
        f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
        cv_scores.append(f1)
        logger.info(f"  Fold {fold}: F1-macro={f1:.4f}")

    logger.info(f"[CV] F1-macro 均值={np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # ---------- 5. 全训练集训练最终模型 ----------
    logger.info("[训练] 全训练集训练最终模型...")
    final_model = XGBoostClassifierChain(
        n_labels=n_labels,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        scale_pos_weights=scale_pos_weights,
        random_state=random_state,
    )
    final_model.fit(X_train.values, y_train.values)

    # ---------- 6. 在验证子集上优化阈值 ----------
    logger.info("[阈值] 搜索最优决策阈值...")
    # 用训练集的 10% 作为验证集来调阈值
    X_thresh, X_thresh_rest, y_thresh, y_thresh_rest = train_test_split(
        X_train, y_train, test_size=0.9, random_state=random_state
    )
    proba_thresh = final_model.predict_proba(X_thresh.values)
    best_thresholds = find_best_thresholds(y_thresh.values, proba_thresh, metric="f1")

    logger.info("[阈值] 最优阈值（vs 默认 0.50）:")
    for j, name in enumerate(label_names):
        logger.info(
            f"  {name}: {best_thresholds[j]:.2f} (delta={best_thresholds[j]-0.5:+.2f})"
        )

    # ---------- 7. 测试集评估 ----------
    logger.info("[评估] 测试集评估...")
    proba_test = final_model.predict_proba(X_test.values)
    results = evaluate_multilabel(
        y_test.values, proba_test, label_names, thresholds=best_thresholds
    )

    _print_results(results, label_names)

    # ---------- 8. 保存 ----------
    if save_model:
        import joblib
        model_path = MODEL_DIR / "xgboost_chain_v1.joblib"
        save_obj = {
            "model": final_model,
            "thresholds": best_thresholds,
            "label_names": label_names,
            "feature_names": list(X_train.columns),
            "scale_pos_weights": scale_pos_weights,
            "norm_stats": norm_stats,
            "metrics": results,
        }
        joblib.dump(save_obj, model_path)
        logger.info(f"[保存] 模型 → {model_path}")

        # 保存评估报告（JSON）
        report_path = _RESULT_DIR / "stage03_evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"[保存] 评估报告 → {report_path}")

    # ---------- 9. 可视化 ----------
    if plot:
        _plot_results(results, label_names, y_test.values, proba_test)

    return results


def _print_results(results: Dict, label_names: List[str]) -> None:
    """打印评估结果摘要。"""
    print("\n" + "=" * 70)
    print("多标签分类评估结果（XGBoost ClassifierChain）")
    print("=" * 70)
    print(f"  Subset Accuracy : {results['subset_accuracy']:.4f}")
    print(f"  Hamming Loss    : {results['hamming_loss']:.4f}")
    print(f"  F1-macro        : {results['f1_macro']:.4f}")
    print(f"  F1-micro        : {results['f1_micro']:.4f}")
    print(f"  Precision-macro : {results['precision_macro']:.4f}")
    print(f"  Recall-macro    : {results['recall_macro']:.4f}")
    print(f"  AUC-ROC (macro) : {results['auc_roc_macro']:.4f}")
    print(f"  AUC-PR  (macro) : {results['auc_pr_macro']:.4f}")
    print(f"  Label Ranking AP: {results['label_ranking_ap']:.4f}")
    print()

    # 逐标签详细指标
    print(f"{'标签':<12} {'正样本':>6} {'F1':>7} {'Prec':>7} {'Rec':>7} {'阈值':>6}")
    print("-" * 55)
    for name in label_names:
        d = results["per_label"][name]
        marker = " *" if d["f1"] >= 0.5 else ""
        print(
            f"  {name:<10} {d['support']:>6} {d['f1']:>7.4f}"
            f" {d['precision']:>7.4f} {d['recall']:>7.4f} {d['threshold']:>6.2f}{marker}"
        )
    print("  * = F1 >= 0.5")
    print("=" * 70)


def _plot_results(
    results: Dict,
    label_names: List[str],
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    """
    生成 3 张图：
        1. 逐标签 F1 / Precision / Recall 柱状图
        2. AUC-PR 分布图
        3. 最优阈值 vs 默认 0.5 对比图
    """
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    n_labels = len(label_names)

    # --- 图 1：逐标签 F1/Precision/Recall ---
    f1s = [results["per_label"][n]["f1"] for n in label_names]
    precs = [results["per_label"][n]["precision"] for n in label_names]
    recs = [results["per_label"][n]["recall"] for n in label_names]
    threshs = [results["per_label"][n]["threshold"] for n in label_names]
    supports = [results["per_label"][n]["support"] for n in label_names]

    x = np.arange(n_labels)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.bar(x - width, f1s, width, label="F1")
    ax.bar(x, precs, width, label="Precision")
    ax.bar(x + width, recs, width, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Label Metrics (F1 / Precision / Recall)")
    ax.legend()
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="F1=0.5")

    # --- 图 2：阈值对比 ---
    ax2 = axes[1]
    ax2.bar(x - width / 2, threshs, width, label="最优阈值", color="steelblue")
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="默认 0.50")
    ax2.set_xticks(x)
    ax2.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Threshold")
    ax2.set_title("Per-Label Optimal Threshold (PR 曲线搜索)")
    ax2.legend()

    plt.tight_layout()
    fig_path = _RESULT_DIR / "stage03_per_label_metrics.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[可视化] 保存 → {fig_path}")

    # --- 图 3：PR 曲线（抽样 5 个标签） ---
    fig, ax = plt.subplots(figsize=(10, 7))
    from sklearn.metrics import precision_recall_curve

    # 选 5 个有代表性的标签
    indices = [0, 4, 8, 14, 18]  # 高血压, 糖尿病, 痛风, 胆结石, 蛀牙
    colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors):
        name = label_names[idx]
        prec, rec, _ = precision_recall_curve(y_true[:, idx], y_proba[:, idx])
        ap = average_precision_score(y_true[:, idx], y_proba[:, idx])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name} (AP={ap:.3f}, n={int(supports[idx])})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves (Selected Labels)")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    pr_path = _RESULT_DIR / "stage03_pr_curves.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[可视化] 保存 → {pr_path}")


# ============================================================
# 新病人预测
# ============================================================


def predict_new(
    user_input: pd.DataFrame,
    model_path: Optional[Path] = None,
) -> Dict:
    """
    对新病人做多标签预测。

    参数：
        user_input: 预处理后的单行 DataFrame（与训练特征列一致）
        model_path: 模型文件路径（None 用默认路径）

    返回：
        {
            "diseases": [(病名, 概率), ...],   # 按概率降序
            "binary_predictions": {病名: 0/1},
            "thresholds_used": {病名: 阈值},
        }
    """
    import joblib

    if model_path is None:
        model_path = MODEL_DIR / "xgboost_chain_v1.joblib"

    obj = joblib.load(model_path)
    model = obj["model"]
    thresholds = obj["thresholds"]
    label_names = obj["label_names"]

    proba = model.predict_proba(user_input.values)[0]
    ranking = sorted(
        zip(label_names, proba),
        key=lambda x: x[1],
        reverse=True,
    )

    binary = {}
    for j, (name, p) in enumerate(ranking):
        binary[name] = 1 if p >= thresholds[j] else 0

    return {
        "diseases": [(d, round(float(p), 4)) for d, p in ranking],
        "binary_predictions": binary,
        "thresholds_used": dict(zip(label_names, [round(float(t), 2) for t in thresholds])),
    }


# ============================================================
# 入口
# ============================================================


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("阶段③：多标签疾病预测 — XGBoost ClassifierChain")
    print("=" * 70)

    results = train_and_evaluate(
        data_path=DATA_PATH,
        test_size=0.2,
        random_state=42,
        n_folds=5,
        save_model=True,
        plot=True,
    )

    # 演示预测
    print("\n" + "=" * 70)
    print("演示：加载模型并预测新样本")
    print("=" * 70)

    # 用测试集第一条做演示
    df = load_data(DATA_PATH)
    X, _ = split_X_y(df)
    sample = X.iloc[:1]

    pred = predict_new(sample)
    print("\nTop-5 预测疾病（按概率）:")
    for disease, prob in pred["diseases"][:5]:
        bin_val = pred["binary_predictions"][disease]
        thresh = pred["thresholds_used"][disease]
        flag = " ← 预测患病" if bin_val else ""
        print(f"  {disease:<10} 概率={prob:.4f}  阈值={thresh}  决策={bin_val}{flag}")


if __name__ == "__main__":
    main()
