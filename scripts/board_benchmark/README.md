# 真板测试复现

这些脚本不烧录、不重启、不读取 NAND。先停止界面采集，等待当前录音任务退出，保持供电。若 IDLE 后仍有 arecord，不要开始并发录音；先调查残留进程。本次采集异常见 [板级结果](../../results/board-20261020/README.md)。旧版 ADB 有长命令限制，脚本使用短路径并检查返回状态。

需要 Python 3 和已安装的 ADB。`ADB` 为可执行文件路径，`ANDROID_SERIAL` 默认 `1234`。例如 PowerShell：

```powershell
$env:ADB='D:\path\to\adb.exe'
$env:ANDROID_SERIAL='1234'
```

准备本地数据目录 `/path/to/prepared`，含 `manifest.json` 和 `b/000.wav` … `b/064.wav`。manifest 使用 [冻结输入清单](../../results/board-20261020/input-manifest.json)，WAV 顺序按其 records 顺序，必须与每条 SHA256 相符。音频不在 Git 中分发；不能换音频却沿用本次标签或统计。原始标签听审、音频转换和许可记录继续保留在本地数据工作区。

```sh
python scripts/board_benchmark/prepare.py --data /path/to/prepared --output /path/to/fresh-run --image-sha256 c8d1fb3c6a552ec03e8a3349b1901ef120edd5b4f1a62320993852f973dfe5dc
python scripts/board_benchmark/evaluate.py --manifest /path/to/prepared/manifest.json --output /path/to/fresh-run
python scripts/board_benchmark/fixed_latency.py --wav /path/to/three-second.wav --output /path/to/fresh-latency
```

prepare 首次需要板上 `/data/c20` 不存在；创建后用本地 owned-test-directory 标记识别归属，不会覆盖未知目录。已有当前测试目录时，应复用原测试目录的标记与 roundtrip，或在人工检查后修改专用路径，不要自动删除已有数据。image-sha256 是操作者声明的烧录镜像身份，脚本只核验运行模型 SHA、AP 分区大小和音频往返字节，不读取 NAND、不声称回读镜像一致。`flash-success.json` 为兼容历史文件名，内容明确写明该边界。

fixed_latency 输入要求三秒、16 kHz、单声道 PCM16，预热 3 次再测 20 次。板上需已有 prepare 创建的 `/data/c20`。输出目录必须全新，保留每次记录。采集准确率、实际告警时延、功耗及长期稳定性不在该 WAV 脚本覆盖范围内。

本次 ARM 状态大小核算命令（已在 Ubuntu SDK 实际执行等价探针）：

```sh
arm-none-eabi-gcc -mcpu=cortex-a7 -mthumb -I app/audiodetect -c scripts/board_benchmark/state_size.c -o /tmp/cnn-state-size.o
arm-none-eabi-nm -S --size-sort /tmp/cnn-state-size.o
# cnn_v2_state_bytes size: 00005238 = 21048 B
```

## 主机资源对照

使用已批准安装的 CPU PyTorch、NumPy、scikit-learn、librosa、torchlibrosa 环境；此处不自动安装软件。PANNs 来源和权重需经 `training/cnn_v2/download_panns.py` 核验，嵌入缓存来自原 baseline 的 390 条训练 + 65 条外部录音。脚本仅从前 390 条特征重拟合最终线性 SVM，检查后 65 条预测与原结果相同；不训练 CNN，拟合时间不计入推理时间。

```sh
python training/cnn_v2/benchmark_resources.py --model cnn --audio /path/to/at-least-10s.wav --output /path/to/cnn-host.json
python training/cnn_v2/benchmark_resources.py --model panns --audio /path/to/at-least-10s.wav --panns-dir /path/to/artifacts/panns --embedding-cache /path/to/artifacts/baselines/panns-embeddings.npz --output /path/to/panns-host.json
```

每模型独立新进程、单线程、batch 1、3 次预热和 10 次测量。输入在内存，完整路径包含模型规定的前端/重采样与推理，不含 WAV 读取、模型加载、SVM 拟合和传输。CNN prepared-input 路径已含前端后的特征；PANNs prepared-input 路径为 32 kHz PCM，网络本身含 STFT/Mel，不能把两者的 core 数值当相同前处理口径。完整 PCM 路径才是本次主要时间对照。JSON 附带原始时间、模型/输入哈希、线程设置、参数与 MAC 范围以及主机进程工作集。

验收公开统计和模型身份：

```sh
python scripts/board_benchmark/verify_published_results.py
sh scripts/verify_submission.sh
```
