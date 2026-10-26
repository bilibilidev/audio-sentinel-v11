# Tiny MLP 固定配方实验与 Linux C 验收

本轮为正式92→96→5模型重做10组控制变量消融、标准SVM/RF/PANNs基线、三档噪声和实际Ubuntu C验收。不继续CNN搜索，不改正式权重或0.80门限。结果见[`results/mlp-study-20261026`](../../results/mlp-study-20261026/README.md)。

训练使用已有Windows CPU Python3.12.14、PyTorch2.14.1+cpu、NumPy2.3.5及已获准安装的scikit-learn/SciPy/librosa等；各版本在协议里记录。Linux仅需要现有Python3标准库、GCC、数学库及已有ASan/UBSan，无需在来宾安装PyTorch/NumPy。

## 输入和协议

- `AUDIT/mlp-training/`包含原390条的干净/同训练折增强特征及输入审计；由`training/retrain_mlp_audit.py`生成，缓存哈希锁定，不加载旧权重初始化。
- `INPUTS`包含`manifest.json`、`esc-manifest.json`、`keyword-manifest.json`及`inputs/`、`esc-inputs/`、`keyword-inputs/`，所有方法使用同一套已校验PCM16文件。原始数据不发布。
- `PANNs`为此前已下载的官方Cnn6、对应`models.py`/`pytorch_utils.py`和`provenance.json`，不重复下载；已冻结AudioSet权重只提特征。
- 全部变体固定60epochs，调度周期240，与正式训练一致；相同种子和顺序，按官方折/源ID保留，尺度和归一化只拟合训练折。每个变体训练五折与一个独立全量模型。全量full逐项核验与正式模型参数一致。
- 特征子集为40/52/66维真实网络输入；代码仍计算完整92维C前端再映射，不声称前端省时。
- 外部65条是已见开发集，含20条婴儿，不用于梯度或基线超参数拟合，不是未见盲测。固定配方成绩与2026-10-24的内层选型成绩分开。

## 训练和基线（已有CPU环境）

```sh
python training/mlp_study/train_ablations.py --audit AUDIT --output RUN/ablations
python training/mlp_study/baselines.py --inputs INPUTS --panns PANNs --output RUN/baselines
python training/mlp_study/resources.py --panns PANNs --output RUN/resources.json
```

`train_ablations.py`导出每个模型NPZ、C头文件和收据。结果目录保存全部60个NPZ；SVM/RF/PANNs另保存18个本地估计器及哈希，不在仓库分发pickle或官方checkpoint。所有候选和网格在评分前固定；没有用外部成绩再次替换正式模型。

## 实际 Linux C

```sh
# 在有NumPy的准备端生成冻结模型/头文件/受控噪声输入目录。
python training/mlp_study/prepare_linux.py --inputs INPUTS --study RUN/ablations --output STAGED

# 将本仓库及STAGED中的inputs、study、reference、exports复制到Ubuntu独立RUN目录。
# 下列命令在Ubuntu执行，路径按本机位置替换；不覆盖旧SDK。
python3 REPO/training/mlp_study/linux_suite.py --repo REPO --inputs RUN/inputs --reference RUN/reference --exports RUN/exports --study RUN/study --output RUN/linux-results
python3 REPO/training/mlp_study/verify_aux.py --binary RUN/linux-results/default/audiodetect --inputs RUN/inputs --reference RUN/reference/results.json --output RUN/linux-results/auxiliary.json
```

Linux套件执行默认构建、自检、正式旧选型六模型复测、新固定配方/全部消融、20/10/5dB保留折噪声、PCM/WAV、读取块、非法输入、ASan/UBSan及计时/RSS/ELF统计。关键词只用合成回归夹具，HTTP/MQTT只连接127.0.0.1。

将Linux结果复制回已有NumPy环境后，重新进行同特征网络数值核对：

```sh
python training/mlp_study/verify_parity.py --linux RUN/linux-results --inputs INPUTS --study RUN/ablations --formal model/mlp-optimized-20261024 --output RUN/parity.json
```

`prepare_linux.py`中的高斯噪声生成方法/种子与实际实验一致。首次执行时修正过C特征索引数组声明顺序，未改任何NPZ；执行源码、头文件修正及哈希均保留在结果证据中。浮点训练跨平台/库版本不保证文件逐字节一致，应检查参数和数值一致性。

## 验收边界

本轮C运行在Ubuntu22.04 x86_64；训练/传统基线在Windows CPU，不能混称Linux Python训练。热缓存WAV函数耗时、CLI进程墙钟和声音至告警时延分开；RSS不是板端RAM，ELF不是固件。未测试完整OpenVela/goldfish、真实板、麦克风、功率或1h/24h稳定性。MLP仍按一秒窗口/三秒UI文件运行，不是CNN的100ms连续流。
