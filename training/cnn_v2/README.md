# CNN v2 实验入口

使用 CPU PyTorch；依赖版本见 `requirements.txt`。原始 ESC-50 和外部录音不纳入 Git。

```sh
export ESC50_DATASET=/absolute/path/to/ESC-50
export CNN_V2_RUN_DIR=/absolute/path/to/new-experiment
python training/cnn_v2/prepare_features.py
python training/cnn_v2/train_suite.py
```

同一份 `training/cnn/esc50-selection.json` 固定 390 条、官方五折和原始文件哈希。每个外层保留折的下一折作为内层验证，其余三折训练。归一化只拟合对应训练折；内层指标选择 epoch 和配置，再在四个训练折重训。外部录音不被训练程序读取。最终模型使用全部 390 条训练，配置由内层分数决定；不能在这 390 条上评价最终模型并称作保留折准确率。

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
