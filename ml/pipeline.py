"""
ml/pipeline.py — 统一机器学习管道 (新版)
==========================================

编排 stage01 → stage02 → stage03 → stage04 的全流程：

    stage01_load:        加载 data/data_ill_final.csv
    stage02_preprocess:  清洗 + Z-score 标准化 + 归一化统计量
    stage03_ML_train:    XGBoost + Classifier Chain 训练模型
    stage04_predict:     推理接口 (前端 / 批量)

使用方式
--------

1. 完整训练：
    >>> from ml.pipeline import MLPipeline
    >>> pipe = MLPipeline()
    >>> pipe.run()                       # 跑完所有 stage 并保存模型

2. 仅训练：
    >>> pipe = MLPipeline()
    >>> pipe.train_only()

3. 加载已训练模型做推理：
    >>> pipe = MLPipeline.load()
    >>> result = pipe.predict(user_dict, history={"hypertension": 1}, top_n=3)

CLI
---
    python -m ml.pipeline run           # 训练并保存模型
    python -m ml.pipeline predict       # 用内置示例做一次预测
    python -m ml.pipeline check         # 自检 (数据 / 统计量 / 模型)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 把项目根目录加入 path，使 ml.* 子包可以正确解析
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ============================================================
# 路径常量
# ============================================================

PROJECT_ROOT: Path = _PROJECT_ROOT
ML_DIR: Path = PROJECT_ROOT / "ml"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODEL_DIR: Path = ML_DIR / "models"
INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"

DEFAULT_DATA_CSV: Path = DATA_DIR / "data_ill_final.csv"
DEFAULT_MODEL_PATH: Path = MODEL_DIR / "xgboost_chain_v1.joblib"
DEFAULT_NORM_STATS_PATH: Path = INTERMEDIATE_DIR / "stage02_norm_stats.json"


# ============================================================
# 数据类
# ============================================================

@dataclass
class PipelineBundle:
    """训练产物 (供推理使用)"""
    model_path: Path
    norm_stats_path: Path
    label_names: List[str]
    feature_names: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    n_train: int = 0
    n_test: int = 0


# ============================================================
# MLPipeline
# ============================================================

class MLPipeline:
    """
    新版统一管道：串联 stage01 ~ stage04

    Attributes
    ----------
    data_csv : Path
        训练数据 CSV 路径
    model_path : Path
        模型保存 / 读取路径
    norm_stats_path : Path
        归一化统计量 JSON 路径
    bundle : PipelineBundle | None
        训练完成后填充
    """

    def __init__(
        self,
        data_csv: Optional[Path] = None,
        model_path: Optional[Path] = None,
        norm_stats_path: Optional[Path] = None,
    ):
        self.data_csv = Path(data_csv) if data_csv else DEFAULT_DATA_CSV
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.norm_stats_path = Path(norm_stats_path) if norm_stats_path else DEFAULT_NORM_STATS_PATH

        self.df_raw: Optional[pd.DataFrame] = None
        self.bundle: Optional[PipelineBundle] = None

        # 确保输出目录存在
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.norm_stats_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Stage 01 — 加载
    # --------------------------------------------------------

    def stage01_load(self) -> pd.DataFrame:
        """从 CSV 加载原始数据。"""
        from ml.stage01_load import load_data, get_data_info

        if not self.data_csv.exists():
            raise FileNotFoundError(
                f"训练数据不存在: {self.data_csv}\n"
                f"请把 data_ill_final.csv 放到 {self.data_csv.parent}"
            )

        logger.info(f"[pipeline] Stage 01 — 加载数据: {self.data_csv}")
        self.df_raw = load_data(source="csv", path=str(self.data_csv))
        info = get_data_info(self.df_raw)
        logger.info(
            f"[pipeline] 数据规模: {info['n_rows']} 行 × {info['n_cols']} 列"
        )
        return self.df_raw

    # --------------------------------------------------------
    # Stage 02 — 预处理
    # --------------------------------------------------------

    def stage02_preprocess(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """清洗 + Z-score + 保存归一化统计量。"""
        from ml.stage02_preprocess import preprocess

        if df is None:
            if self.df_raw is None:
                self.stage01_load()
            df = self.df_raw
        assert df is not None

        logger.info("[pipeline] Stage 02 — 预处理")
        df_clean = preprocess(df, verbose=False)

        # 归一化统计量由 stage02 内部已保存到 self.norm_stats_path
        # 若要重新生成：可在 stage02_preprocess.py 中覆盖
        if not self.norm_stats_path.exists():
            logger.warning(
                f"[pipeline] 归一化统计量不存在: {self.norm_stats_path}"
                "（stage02 通常会自动生成，请确认 stage02_preprocess.py 写盘逻辑）"
            )
        else:
            logger.info(f"[pipeline] 归一化统计量已就绪: {self.norm_stats_path}")

        return df_clean

    # --------------------------------------------------------
    # Stage 03 — 训练
    # --------------------------------------------------------

    def stage03_train(
        self,
        df: Optional[pd.DataFrame] = None,
        save_model: bool = True,
    ) -> PipelineBundle:
        """训练 XGBoost Classifier Chain 模型。"""
        from ml.stage03_ML_train import XGBoostClassifierChain  # noqa: F401 触发注册

        if df is None:
            df = self.stage02_preprocess()

        logger.info("[pipeline] Stage 03 — 训练 XGBoost Classifier Chain")
        t0 = time.time()

        # 训练入口直接在 stage03_ML_train 中以 main 形式提供。
        # 这里用 subprocess 触发，方便复用 stage03 的 CLI 参数和报告输出。
        import subprocess
        cmd = [
            sys.executable,
            str(ML_DIR / "stage03_ML_train.py"),
            "--data", str(self.data_csv),
            "--model_out", str(self.model_path),
            "--norm_stats_out", str(self.norm_stats_path),
        ]
        logger.info(f"[pipeline] 启动训练: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(result.stdout)
            logger.error(result.stderr)
            raise RuntimeError(
                f"stage03 训练失败 (exit {result.returncode})"
            )
        elapsed = time.time() - t0
        logger.info(f"[pipeline] 训练完成，耗时 {elapsed:.1f}s")

        # 收集元数据
        label_names: List[str] = []
        feature_names: List[str] = []
        try:
            import joblib
            obj = joblib.load(self.model_path)
            label_names = list(obj.get("label_names", []))
            feature_names = list(obj.get("feature_names", []))
        except Exception as e:
            logger.warning(f"[pipeline] 读取模型元数据失败: {e}")

        self.bundle = PipelineBundle(
            model_path=self.model_path,
            norm_stats_path=self.norm_stats_path,
            label_names=label_names,
            feature_names=feature_names,
            elapsed_s=round(elapsed, 2),
        )
        return self.bundle

    # --------------------------------------------------------
    # Stage 04 — 推理
    # --------------------------------------------------------

    def predict(
        self,
        user_dict: Dict[str, Any],
        history: Optional[Dict[str, Any]] = None,
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """对单条用户数据做疾病预测。"""
        self._ensure_chain_class_in_main()
        from ml.stage04_predict import predict as _predict
        return _predict(
            user_dict=user_dict,
            history=history,
            top_n=top_n,
        )

    def predict_batch(
        self,
        user_dicts: List[Dict[str, Any]],
        history_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        self._ensure_chain_class_in_main()
        from ml.stage04_predict import predict_batch as _pb
        return _pb(
            user_dicts=user_dicts,
            history_list=history_list,
            top_n=top_n,
        )

    # --------------------------------------------------------
    # 训练入口
    # --------------------------------------------------------

    def train_only(self) -> PipelineBundle:
        """执行 Stage 01 → 02 → 03，不做预测。"""
        df_raw = self.stage01_load()
        df_clean = self.stage02_preprocess(df_raw)
        return self.stage03_train(df_clean)

    def run(self) -> PipelineBundle:
        """完整流程：加载 → 预处理 → 训练 → 自检。"""
        logger.info("=" * 60)
        logger.info("MLPipeline 启动 (新版 stage01-04)")
        logger.info(f"数据源: {self.data_csv}")
        logger.info(f"模型输出: {self.model_path}")
        logger.info("=" * 60)
        bundle = self.train_only()
        logger.info("=" * 60)
        logger.info(f"训练完成 — {len(bundle.label_names)} 标签, {bundle.elapsed_s}s")
        logger.info("=" * 60)
        return bundle

    # --------------------------------------------------------
    # 类方法：从已训练产物加载
    # --------------------------------------------------------

    @staticmethod
    def _ensure_chain_class_in_main() -> None:
        """让 pickle 能找到训练时记录的 __main__.XGBoostClassifierChain。"""
        import sys
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and "XGBoostClassifierChain" not in getattr(main_mod, "__dict__", {}):
            from ml.stage03_ML_train import XGBoostClassifierChain
            setattr(main_mod, "XGBoostClassifierChain", XGBoostClassifierChain)

    @classmethod
    def load(
        cls,
        model_path: Optional[Path] = None,
        norm_stats_path: Optional[Path] = None,
    ) -> "MLPipeline":
        """从已有模型文件构造一个可直接 predict() 的实例。"""
        cls._ensure_chain_class_in_main()

        instance = cls.__new__(cls)
        instance.data_csv = DEFAULT_DATA_CSV
        instance.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        instance.norm_stats_path = (
            Path(norm_stats_path) if norm_stats_path else DEFAULT_NORM_STATS_PATH
        )
        instance.df_raw = None

        # 加载元数据
        try:
            import joblib
            obj = joblib.load(instance.model_path)
            instance.bundle = PipelineBundle(
                model_path=instance.model_path,
                norm_stats_path=instance.norm_stats_path,
                label_names=list(obj.get("label_names", [])),
                feature_names=list(obj.get("feature_names", [])),
            )
        except FileNotFoundError:
            logger.warning(
                f"[pipeline.load] 模型文件不存在: {instance.model_path}，"
                "请先调用 MLPipeline().run() 训练"
            )
            instance.bundle = None
        return instance

    # --------------------------------------------------------
    # 自检
    # --------------------------------------------------------

    def check(self) -> Dict[str, Any]:
        """自检：数据 / 归一化统计量 / 模型是否齐全。"""
        return {
            "data_csv": {
                "path": str(self.data_csv),
                "exists": self.data_csv.exists(),
            },
            "norm_stats": {
                "path": str(self.norm_stats_path),
                "exists": self.norm_stats_path.exists(),
            },
            "model": {
                "path": str(self.model_path),
                "exists": self.model_path.exists(),
            },
            "bundle_loaded": self.bundle is not None,
            "labels": self.bundle.label_names if self.bundle else None,
        }


# ============================================================
# CLI
# ============================================================

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ml.pipeline",
        description="统一机器学习管道 (stage01-04)  |  无参数 = 自检 (check)",
        epilog=(
            "用法示例:\n"
            "  python ml/pipeline.py            # 等同 check\n"
            "  python ml/pipeline.py check      # 自检\n"
            "  python ml/pipeline.py predict    # 用内置示例做一次预测\n"
            "  python ml/pipeline.py run        # 完整训练\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # required=False: 不传子命令时走兜底（默认 check）
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("run", help="完整训练 (Stage 01 → 03)")
    sub.add_parser("check", help="自检: 数据 / 统计量 / 模型")
    sub.add_parser("predict", help="用内置示例做一次预测")

    args = parser.parse_args(argv)

    _setup_logging()

    # 兜底: 无子命令 → 自检 (PyCharm 绿三角直接点运行也能跑通)
    if args.cmd is None:
        logger.info("[pipeline] 未指定子命令，默认执行 check（自检）")
        args.cmd = "check"

    if args.cmd == "run":
        MLPipeline().run()
        return 0

    if args.cmd == "check":
        pipe = MLPipeline.load()
        info = pipe.check()
        print("\n=== 自检 ===")
        for k, v in info.items():
            print(f"{k}: {v}")
        return 0

    if args.cmd == "predict":
        from ml.stage04_predict import predict as _predict
        sample = {
            "gender": "男",
            "sbp": 162, "dbp": 100, "pulse": 82,
            "weight": 82, "height": 175, "bmi": 26.8,
            "cholesterol": 240,
            "glucose": 105, "triglycerides": 195,
            "exercise": False, "sleep_hours": 5.5,
            "snoring_freq": 3, "salt_intake": 3,
        }
        result = _predict(sample, history={"hypertension": 1}, top_n=3)
        print(f"\n最可能疾病: {result['top_disease']}")
        print(f"置信度: {result['confidence']:.2%}")
        for item in result["top_n"]:
            print(f"  #{item['rank']} {item['disease']:<25} {item['probability']:.4f}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())