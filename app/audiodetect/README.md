# audiodetect：OpenVela 端侧应用

本目录是 Audio Sentinel 的 OpenVela/NuttX 应用源码。当前默认模型为 CNN v2；`audio-sentinel-v11.xml` 将它映射到工作区的 `apps/audiodetect`。

## 当前实现

- 五类音频事件：background、cough、glass_break、baby_cry、dog_bark。
- 因果时序 CNN，25 ms 帧、10 ms 帧移；特征、通道、dilation 与聚合方式由导出模型配置固定。
- INT8 权重存储、浮点卷积；不是全整型运算。
- WAV 命令行和 UI 共用 `audio_classifier.c` 与 `cnn_stream_v2.c`。
- 三秒麦克风文件采集，每个文件内部连续推理；CLI 读取完整文件，不截断为一秒或三秒。
- 不调用旧 MLP 的静音强制改判和窗口投票；告警置信度仍为 `0.80`，未宣称实时报警校准完成。
- LVGL触摸界面、自动启动偏好、事件计数和最近三条事件。
- WAV命令行推理、噪声演示、合成模板关键词识别和可选HTTP/MQTT上报代码。

## 主要文件

| 文件 | 作用 |
|---|---|
| `audiodetect_main.c` | CNN v2 命令入口、完整 WAV 推理、告警和子命令分发 |
| `audiodetect_ui.c` | LVGL界面、麦克风三秒采集和后台工作线程 |
| `audio_classifier.c` | App 共用的 CNN v2 全片分类封装，状态分配在堆上 |
| `cnn_stream_v2.c`、`cnn_stream_v2.h` | 因果 DSP、卷积缓存、在线 EMA 与完整文件聚合 |
| `cnn_model_config_v2.h`、`cnn_weights_v2.h` | 最终模型尺寸、SHA256 与生成权重 |
| `audio_classifier_mlp.c`、`feature_extractor.c`、`model_infer*.c`、`model_weights*.h` | 保留的原 v11 MLP 回退，不参与新默认 App 构建 |
| `wav_reader.c` | 16 kHz、16-bit、单声道PCM WAV读取 |
| `keyword_detect.c` | 保留的 MFCC+DTW 关键词路径与旧 MLP 的 MFCC 统计工具 |
| `keyword_templates.h` | 合成语音生成的关键词模板，不是真实录音训练数据 |
| `alarm_report.c` | 可选HTTP/MQTT告警代码 |
| `Kconfig`、`Makefile`、`CMakeLists.txt` | OpenVela构建集成 |

## 运行

启用以下配置：

```text
CONFIG_AE_AUDIODETECT=y
CONFIG_AE_AUDIODETECT_UI=y
CONFIG_AE_AUDIODETECT_UI_AUTOSTART=y
CONFIG_GRAPHICS_LVGL=y
```

命令示例：

```text
nsh> audiodetect ui
nsh> audiodetect /data/test.wav --int8 --debug
nsh> audiodetect robust /data/test.wav
nsh> audiodetect keyword /data/keyword.wav
```

实时UI使用 `arecord -D default -r16000 -f16 -c1 -d3 /data/aed_live.wav`。它不是无间隙流式PCM管线；边界和未验证项请见仓库根目录README。

Linux 主机：

```sh
sh scripts/build_cnn_v2_app.sh
out/cnn-v2/audiodetect input16k.wav --debug
sh scripts/build_cnn_v2_stream.sh
out/cnn-v2/cnn_stream --live input16k.wav
# PCM16 单声道 16 kHz 管线：
out/cnn-v2/cnn_stream --stdin-raw < input16k.pcm
```

`--int8` 为兼容参数，新默认 App 一直使用生成的 INT8 权重。旧 MLP 对照入口为 `scripts/build_robust_c_simulator.sh`；CNN v1 独立入口继续保留。当前模型 SHA256 会在 CLI 中输出，避免误把旧二进制当作新版。

关键词识别仅为命令行原型，模板来自合成语音，未接入实时UI、未进行真实录音独立准确率验证。
