# CNN v2 从零训练候选

每个模型均从随机权重训练 80 epochs，不载入 CNN v1/v2、教师或优化器检查点。
fold_1…fold_5 分别排除 ESC-50 对应官方折；final_all_esc50 使用全部 390 条重新独立训练。
原 `model/cnn-v2`、App 默认权重及开发板固件保持不变。

运行候选 App（仓库根目录，现有 Ubuntu）：

```sh
MODEL_DIR="$PWD/model/cnn-v2-scratch" OUT_DIR="$PWD/out/cnn-v2-scratch" sh scripts/build_cnn_v2_app.sh
MODEL_DIR="$PWD/model/cnn-v2-scratch" OUT_DIR="$PWD/out/cnn-v2-scratch" sh scripts/build_cnn_v2_stream.sh
out/cnn-v2-scratch/audiodetect input.wav --debug
out/cnn-v2-scratch/cnn_stream --live --chunk 160 input.wav
```

INT8 仅指权重/偏置存储，DSP 与卷积仍为浮点。结果见 `results/cnn-v2-scratch`。
