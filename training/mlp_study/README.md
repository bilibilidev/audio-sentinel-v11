# Tiny MLP 训练与实验复现

本目录提供当前正式配方及十组控制变量实验。正式模型为92→96→5，使用asinh40变换、dropout 0.15、AdamW学习率0.002、weight decay 0.001及label smoothing 0.02，固定训练60轮，余弦调度周期240轮。初始化、批次次序和增强种子在协议中记录。

## 特征与训练

在已有CPU PyTorch、NumPy环境中准备ESC-50数据。`esc50-selection.json`固定390条录音、来源、折号和原始哈希。下列目录为本地路径占位符：

```bash
python training/mlp_study/prepare_features.py --esc50 ESC50 --output RUN/cache/mlp-training
python training/mlp_study/train_ablations.py --features-root RUN/cache --output RUN/study
```

特征缓存包含每条录音的五个一秒窗口和四份同折背景增强视图。训练折拟合归一化及asinh尺度；每个变体训练五个保留折模型和一个全量模型，共60个NPZ。每折和全量模型均从随机初始化开始。全量full参数与正式模型逐项核对。

## 导出与基线

```bash
python training/mlp_study/export_model.py model/mlp-optimized-20261024/final_all_esc50.npz RUN/export
python training/mlp_study/baselines.py --inputs INPUTS --panns PANNs --output RUN/baselines
python training/mlp_study/resources.py --panns PANNs --output RUN/resources.json
```

导出保留训练拟合的归一化及asinh尺度。MFCC+SVM、MFCC+RF使用训练折内部选参；PANNs Cnn6采用已有AudioSet预训练冻结特征和线性SVM。原始录音、预训练checkpoint和本地joblib模型按其来源条款单独管理。

## Ubuntu C 验证

准备端使用NumPy生成模型头文件和固定种子的受控噪声输入，然后将源码及STAGED目录复制到Ubuntu：

```bash
python training/mlp_study/prepare_linux.py --inputs INPUTS --study RUN/study --output STAGED
python3 REPO/training/mlp_study/linux_suite.py --repo REPO --inputs STAGED/inputs --study STAGED/study --output RUN/linux
python3 REPO/training/mlp_study/verify_aux.py --binary RUN/linux/default/audiodetect --inputs STAGED/inputs --output RUN/linux/auxiliary.json
```

Linux套件使用标准库与GCC，构建正式App和60个实验模型，执行五折、外部WAV、噪声、PCM/WAV、读取分块、异常输入、ASan/UBSan和计时。原始训练数据不需要在公开Git中分发。

将Linux输出带回NumPy环境后进行同特征网络核对：

```bash
python training/mlp_study/verify_parity.py --linux RUN/linux --inputs STAGED/inputs --study STAGED/study --output RUN/parity.json
```

训练由CPU Python完成，系统实现验证由Ubuntu C完成。外部65条为已见开发集；模拟器、开发板、现场录音和功耗分别进行测量。实验结果见[结果表](../../results/mlp-study-20261026/README.md)。
