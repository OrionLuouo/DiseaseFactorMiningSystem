"""
ml/ — 机器学习模块
==================
子模块：
    config_loader  — 配置加载（读取 ml/config.yaml）
    stage01_load   — 数据加载（支持本地 / HDFS / Spark）
    preprocess_config — 可配置的数据预处理管道
    pipeline       — 统一 ML 管道（训练 / 评估 / 预测）
    stage02_preprocess — 旧版预处理（保留向后兼容）
    stage05_train  — 旧版训练流水线（保留向后兼容）
    gen_reference_samples — 参照样本生成
"""
