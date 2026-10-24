# 2026-10-24 开发优化实验

正式采用单个Tiny MLP，CNN搜索按用户要求停止。训练均从随机权重开始；不把外部65条用于梯度训练，保留失败候选和中断状态。指标边界和每折数据预算见`results/mlp-optimized-20261024`。原始数据/缓存不随代码发布。

## 复现顺序

在既有CPU PyTorch/NumPy/SciPy环境中运行，依赖版本见发布的各`protocol.json`。数据位置用参数传入，不需要购买服务器。

1. 使用已有`training/retrain_mlp_audit.py`生成`AUDIT/mlp-training/`的干净/增强缓存和`input-audit.json`；这是准备缓存和上一版基线的流程，不加载其权重作为初始化。
2. `prepare_expanded.py --dataset ESC50 --output EXPANDED`生成扩充背景特征，保留固定390评测集合，排除跨折源ID的额外背景。
3. `search.py --kind mlp --audit AUDIT --cnn-cache CACHE --output RUN/mlp`运行8候选。`CACHE`为早期CNN特征缓存，在MLP模式仅记录其哈希、不用于MLP训练。
4. `expanded_search.py --kind mlp --audit AUDIT --cnn-cache CACHE --expanded-cache EXPANDED/features.npz --output RUN/mlp-expanded`运行3候选。
5. `consolidate.py --root RUN --kind mlp`按内层分数合并，生成`mlp-selected/frozen.json`；不按外层正确数挑配置。
6. `evaluate_selected.py --root RUN --kind mlp --only-esc --inputs INPUTS --output C_ESC --compiler cc`验证实际C五折；外部评估还需冻结保留CNN选择记录后去掉`--only-esc`。历史记录中的`INPUTS`须按相同清单/哈希准备。
7. `export_mlp.py MODEL.npz EXPORT_DIR`重新导出C头文件，包含asinh尺度。正式权重位于`model/mlp-optimized-20261024`；原App共享`mlp_model_infer.c`承担变换和推理。

`AUDIT`、`CACHE`、`EXPANDED`、`RUN`和`INPUTS`均为本地路径占位符。新输出目录不能覆盖已冻结实验。各次实际命令和输入哈希见结果目录；浮点训练可能受PyTorch/平台版本影响，不承诺模型归档逐字节重建。

## 非正式对照

`attention_search.py`、`evaluate_mlp_ensemble.py`为注意力/固定双MLP对照，外部结果不如正式单MLP。其余CNN/PCEN/谱中心化/抗混叠/时间降采样脚本为实验档案，不是正式默认。部分CNN未完成五折，用户停止后不再续跑。`adapters.py`仅作用于实验输出副本，不改保留App的CNN核心。

早期CNN增强缓存存在未抗混叠的16→8k降采样；`prepare_correct_phone.py`为独立修正版。它同时改变增强视图数量/强度，不应把两轮差异全部归因于抗混叠。正式MLP不依赖该CNN增强视图。
