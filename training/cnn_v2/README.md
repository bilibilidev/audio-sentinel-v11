# CNN v2 实验入口

使用 CPU PyTorch；依赖版本见 `requirements.txt`。原始 ESC-50 和外部录音不纳入 Git。

```sh
export ESC50_DATASET=/absolute/path/to/ESC-50
export CNN_V2_RUN_DIR=/absolute/path/to/new-experiment
python training/cnn_v2/prepare_features.py
python training/cnn_v2/train_suite.py
```

同一份 `training/cnn/esc50-selection.json` 固定 390 条、官方五折和原始文件哈希。每个外层保留折的下一折作为内层验证，其余三折训练。归一化只拟合对应训练折；内层指标选择 epoch 和配置，再在四个训练折重训。外部录音不被训练程序读取。最终模型使用全部 390 条训练，配置由内层分数决定；不能在这 390 条上评价最终模型并称作保留折准确率。

第二轮带宽适配（第一轮最终候选在外部集退化后增加，保留第一轮结果）：

```sh
python training/cnn_v2/train_bandwidth.py
```

第二轮固定 80 epochs，内层同时评价干净音频和仅由对应 ESC-50 验证录音生成的电话带宽视图。效用为 `0.4 × (干净宏召回 + 电话宏召回) + 0.1 × (干净婴儿召回 + 电话婴儿召回)`，不读取实际外部录音。电话视图在降至 8 kHz 前用 polyphase 滤波防止混叠，再使用 v11 的线性 8→16 kHz 输入转换。另对照只保留前 19 个 mel 滤波器的低频特征（非零支撑最高 3781.25 Hz）、干净混合特征和干净 26 维 log-mel。逐折按内层选配置，再重训外层模型；最终配置按平均内层效用选择。外部开发集的使用历史须随结果说明，不能将迭代后的分数称作全新盲测。

第三轮从冻结 CNN v1 权重微调，保留均值聚合：

```sh
python training/cnn_v2/finetune_v1.py
```

内层基模型重新在三折上训练，不能拿已见过内层验证折的 v1 四折模型做内层初始化。外层使用对应 v1 四折权重、最终模型使用 v1 全数据权重；保留其训练折归一化。候选以小学习率微调 20 epochs，比较电话带宽、婴儿损失权重与干净训练输入上的冻结教师 KL 约束。内层按干净/合成电话效用选型，实际外部录音仍不参与梯度训练。原 v1 文件不写入，父模型 SHA256 随新模型保存。

`development_select.py` 明确使用外部 65 条作为开发集筛选固定微调候选，随后评价选中固定配方的 ESC-50 官方五折；该结果不是内层选型的嵌套 CV，也不是外部盲测。暂存候选由这一步产生，必须披露开发选型历史。

`distill_panns.py`、`distill_anchor.py` 为额外 AudioSet 先验的学生模型对照。仅使用冻结 PANNs 缓存的前 390 条 ESC 特征，排除后 65 条外部特征；辅助投影头在部署时移除。32/64 通道扩展在微调前保持原 v1 浮点函数；这些试验的外部退化也保留。它们没有替换当前纯 ESC-50 的保守候选。部署 C 导出器按模型生成缓存和暂存区尺寸，宽通道仍须单独执行真实 C 验收。

预先定义的 11 个控制变量对照包含：基础配置、无增强、无电话音质增强、time-only、MFCC-only、混合特征、16/24/32 通道、三种 dilation 和均值聚合。内层效用为 `0.8 × 五类宏召回 + 0.2 × 婴儿召回`。逐个固定变体与逐折内层选型的嵌套 CV 分开记录。每种特征的时序输入不同于传统 baseline 的全片统计输入。

标准 baseline：

```sh
export EXTERNAL_AUDIO_DIR=/absolute/path/to/curation-v3-baby-20261019
python training/cnn_v2/download_panns.py
python training/cnn_v2/baselines.py
```

外部目录须有原始 `manifest.json`、音频和 `review-acceptance-20261019.json`。MFCC+SVM/RF 与轻量 CNN 的有监督训练数据相同；PANNs Cnn6 冻结 512 维嵌入+线性 SVM 用过额外 AudioSet 预训练数据，需单独比较。下载脚本固定官方代码提交和权重 MD5，保存来源与 SHA256。未运行 YAMNet、AST 或收费云 API，不上传录音。

```sh
python training/cnn_v2/export_c.py /path/to/model.npz /path/to/generated
MODEL_DIR=/path/to/generated OUT_DIR=/path/to/build sh scripts/build_cnn_v2_app.sh
MODEL_DIR=/path/to/generated OUT_DIR=/path/to/build sh scripts/build_cnn_v2_stream.sh
```

导出的 INT8 权重在 C 中反量化执行浮点卷积，并非全整型算术。完整 WAV 的聚合由模型配置固定；在线每 100 ms 输出 EMA 分值，两者的准确率不能直接混用。App 与主机入口共用 `app/audiodetect/cnn_stream_v2.c`。

C/Python 数值与流式状态验收（用自有 WAV 生成输入，不在 Git 中发布数据）：

```sh
python training/cnn_v2/prepare_c_check.py model/cnn-v2/final_all_esc50.npz /path/to/esc-clip.wav /path/to/fresh-check
# 将生成目录复制到 Linux，环境需 NumPy 和 C 编译器：
python /path/to/fresh-check/verify.py
```

检查包含逐帧 logits/EMA、完整录音聚合、1/160/997/4096 样本输入块、前缀因果性、短音频/静音及 ASan/UBSan。验证目录保留协议、源文件哈希、命令、编译日志和二进制哈希。
