# CNN v1：独立 Linux C 候选

本次更新在 `b465d17afd43b6a574f5c368ad90ac24d102cbb7` 上新增流式 CNN。原 v11 的 MLP 权重、特征、阈值、合成关键词原型、应用及 R528S3 配置不变。尚未将 CNN 接入完整 OpenVela 模拟器、UI、麦克风或开发板。

## Linux 构建与启动

在仓库根目录运行；仅需要 Linux C11 编译器、数学库，回放辅助脚本使用 Python 3 标准库。运行推理不依赖 PyTorch，也不需要下载数据集或 OpenVela SDK。

```sh
./scripts/verify_submission.sh
./scripts/build_robust_c_simulator.sh    # 保留的原 MLP 入口
./scripts/build_cnn_c_simulator.sh
./out/cnn-simulator/cnn_stream /path/to/mono16k.wav

# 保留状态的 PCM 输入，每 100 ms 输出 JSONL update；EOF 时输出整片 summary。
python3 simulator/cnn/replay_pcm.py /path/to/mono16k.wav \
  | ./out/cnn-simulator/cnn_stream --stdin-raw

# 快速读取文件并打印在线分值，不按现实时间等待。
./out/cnn-simulator/cnn_stream --live /path/to/mono16k.wav
```

C 入口接收 16 kHz、单声道、16-bit PCM WAV；原始管道为 PCM16 little-endian，当前验证平台为 Ubuntu 22.04 x86_64 / GCC 11.4.0。多个 WAV 参数分别重置状态；一个管道保持连续状态。调用 `cnn_feed()` 可传入任意大小的 PCM 块，`cnn_reset()` 开始新流，`cnn_finish()` 只用于输入结束，不能在每个采集块后调用。

25 ms 分析帧、10 ms 步进；26 个 log-mel 频带；五层 24 通道因果 Conv1d，kernel=5、dilation=1/2/4/8/16；5 类输出，共 14,885 参数。完整历史为 125 帧，覆盖约 1.265 秒。状态约 19,376 字节（本次 x86_64 ABI），不含权重和调用栈。

预加重、STFT、卷积历史跨输入块保留。在线分值是 alpha=0.02 的 logits EMA，每 10 帧输出一次；预热阶段带 `warmup=true`。整片分类使用所有帧 logits 的均值后再 softmax。两种输出不能混为一种指标。权重及偏置用 INT8 存储，FFT、归一化和累加仍用 float，并非全整数网络。

## 已完成的准确率验证

| 同一组 45 条外部录音，实际 Linux C | 原 MLP | CNN v1 |
|---|---:|---:|
| 五个 ESC-50 折模型分别评测后的平均值 | 163/225，72.4% | 178/225，79.1% |
| 单个最终部署模型 / 最终候选 | 35/45，77.8% | 40/45，88.9% |

五个折模型都评测同一组 45 条录音，225 次预测并非 225 条独立录音，也不是对外部数据重新做五折训练。原 C 使用保留的三秒事件优先规则并聚合整条录音；CNN 使用连续帧整片均值，因此表中比较的是完整推理流程。各折、逐文件概率、来源和文件哈希见 [外部结果](../results/cnn-v1/external-reviewed-45.json)。

最终候选的背景/咳嗽/玻璃/狗叫召回率分别为 10/13、11/12、8/9、11/11。剩余错误为 `cough-01`、`glass_break-06`、`background-02`、`background-14`、`background-20`。这 45 条 FSD50K 录音经人工确认，未加入训练，但已经看过测试结果，应视为外部开发验证集；最终泛化评估仍需新的未看结果样本。来源 ID 与 ESC-50 不重合、无完全相同音频；未穷尽近重复片段检测。20 条 Donate-a-Cry 候选待听审，未计分，不声称已完成外部五类评测。

ESC-50 官方保留折 INT8：原 MLP 322/390=82.6%，CNN 313/390=80.3%；CNN Float 为 312/390=80.0%。CNN 在这里有退步，因此保留独立候选入口。CNN 的逐折预测见 [保留折结果](../results/cnn-v1/esc50-held-out.json)。

实际 Linux C 与 Python INT8 版本在外部六个模型的最大概率差约 1.97e-6；分块大小、短音频、静音和前缀因果性已核对，ASan/UBSan 通过。预录 WAV 按原速输入管道时能在 EOF 前输出，每 100 ms 更新；首次输出约 0.11 秒，处于预热阶段，不能当作报警延迟。未测量麦克风准确率、事件起止检测率、每小时误报或真板内存/功耗。

## 复核已有外部录音

原音频不随 Git 发布；按 manifest 的路径放置已合法取得的录音，脚本先核对 SHA256，再用原 v11 `audio_utils.py` 转为 16 kHz 单声道。

```sh
python3 training/cnn/evaluate.py \
  --manifest results/cnn-v1/external-reviewed-45.json \
  --audio-root /path/to/curation-v2 \
  --backend c --binary out/cnn-simulator/cnn_stream \
  --output out/cnn-external-c
```

## 训练、导出与数值验证

CPU 训练环境独立安装；本次使用 Python 3.12.14、NumPy 2.3.5、PyTorch 2.14.1+cpu。Linux C 核验只需要 NumPy，不需要在虚拟机安装 PyTorch。

```sh
python3 -m venv .venv-cnn
.venv-cnn/bin/python -m pip install -r training/cnn/requirements.txt
.venv-cnn/bin/python training/cnn/train_causal_cnn.py \
  --esc50-dir /path/to/ESC-50 --output out/cnn-training

# 已提交的最终候选可直接导出；默认构建使用随仓库提交的头文件。
.venv-cnn/bin/python training/cnn/export_to_c.py \
  model/cnn-v1/final_all_esc50.npz out/cnn-export

# 先在有 CPU PyTorch 的机器生成独立参考；可将 fixtures 传到 Linux。
.venv-cnn/bin/python training/cnn/prepare_parity.py --output out/cnn-parity
python3 training/cnn/verify_c.py --fixtures out/cnn-parity --sanitizers
```

训练仅使用 [固定选择清单](../training/cnn/esc50-selection.json) 中的 390 条 ESC-50 原始录音，核对原始音频及 metadata 哈希。每折只用四个训练折的归一化统计，断言源 ID 不跨训练/测试；固定 80 epoch，不从外部集选轮次或阈值。最终候选单独用全部 390 条训练，不能在这些训练录音上计分并称为交叉验证。导出头文件应与所验证 NPZ 对应；更换候选权重后需重新导出、编译和验证。

训练音频、特征缓存、Python 环境和构建产物不入 Git。权重及哈希在 `model/cnn-v1/`，训练配置和已执行的数值/流式证据在 `results/cnn-v1/`。历史依赖未完整锁定，不保证重训权重、二进制或镜像字节一致；原有第三方来源及使用边界见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。
