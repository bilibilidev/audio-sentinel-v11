# 声哨 Audio Sentinel：基于 OpenVela 的端侧音频检测

声哨面向家庭看护、听力辅助和养宠场景，在 R528S3 Gemini S1 开发板上完成麦克风采集、Tiny MLP 推理和 LVGL 显示，识别背景、咳嗽、玻璃破碎、婴儿哭声和狗叫。音频在设备本地处理，配套提供训练、C 部署、板级构建和实验复现代码。

## 核心亮点

1. **五类声音识别。** 正式模型采用 92→96→5 Tiny MLP，从随机初始化训练 60 轮。ESC-50 五类子集官方五折为 **337/390（86.41%）**，高于本轮 MFCC + SVM/RF 基线；外部 65 条录音分类为 **57/65（87.69%）**。
2. **噪声鲁棒性。** 在 20/10/5 dB 高斯噪声下，五折准确率分别为 **77.44% / 72.31% / 68.46%**。`robust` 命令支持单条 WAV 的干净与加噪对比，`demos/robust_demo.sh` 现场生成加噪输入并调用 Linux C 分类。
3. **轻量模型。** 92 维输入由 40 维时域统计和 52 维 MFCC/Delta 组成，时域统计采用 asinh 压缩。模型仅 **9,413 个参数**，INT8 权重/偏置占 **9,413 B**，归一化与 scale 等占 **912 B**；权重/偏置存储约为 float32 的 1/4，推理使用浮点运算。Float 与 INT8 存储版本五折均为 337/390；五秒输入网络为 **46,560 MAC**，Linux C 三秒 WAV 函数处理均值为 **2.482 ms**。
4. **远程告警。** `--report` 支持 HTTP POST `/api/alarm` 和 MQTT PUBLISH `audiodetect/<device>/alarm`，发送 device、event、confidence、severity 和 timestamp。玻璃破碎为 critical，婴儿哭声和狗叫为 warning，咳嗽为 info；**127.0.0.1:8080/1883 本地回环已通过**。
5. **关键词识别。** 独立 MFCC + DTW 命令行路径识别“救命”“帮助”“报警”。使用 13 维 MFCC、26 个 mel 滤波器、512 点 FFT、400/160 样本帧长/帧移及 DTW 阈值 50；合成模板回归 **9 条正例、2 条负例通过**。真实语音准确率待测。
6. **开发板适配。** 集成 OpenVela/NuttX、音频采集、GT911 触摸、ILI9341 小屏、LVGL 和开机启动，提供 START、STOP、AUTO 及事件摘要。R528S3 对外部 65 条 WAV 分类为 **57/65（87.69%）**，与 Linux C 的 **65/65 条预测一致**；三秒 RAM WAV 处理均值为 **84.067 ms**。

## 实验结果

| 模型 | ESC-50 五类官方五折 | 外部开发集（65 条） |
|---|---|---|
| Tiny MLP | 337/390，86.41% | 57/65，87.69% |
| MFCC + SVM | 312/390，80.00% | 39/65，60.00% |
| MFCC + RF | 299/390，76.67% | 38/65，58.46% |
| PANNs Cnn6 + 线性 SVM | 370/390，94.87% | 63/65，96.92% |

ESC-50 成绩统计固定的五类 390 条录音，Tiny MLP 为固定配方五折评估。**外部 65 条录音已听审确认标签，覆盖全部五类，统一计分**；录音来自 FSD50K 与 Donate-a-Cry，未用于梯度训练，但已参与开发选型，属于开发验证集。PANNs 使用额外 AudioSet 预训练，准确率更高；Tiny MLP 侧重端侧存储与计算规模。完整基线、十组消融和噪声结果见[实验说明](docs/EVALUATION.md)。

时间统计包含 WAV 读取、DSP 与推理，录音和 USB 传输另计。完整录音按一秒窗口处理，对全部窗口概率取平均，尾窗补零；UI 每三秒采集一个文件。分类采用 argmax，告警另要求非背景且置信度大于 0.80，分类准确率与告警触发率分别统计。

真板已完成五次麦克风采集和 200 次短时推理循环；NAND 批量写入仍出现 SPI0/YAFFS 故障，板端网络送达、现场麦克风分类准确率、1 h/24 h 稳定性及功耗待测。具体条件见[真板实验](results/board-mlp-20261027/README.md)。

## Linux C 与视频演示

在 Ubuntu 终端编译并推理：

```bash
bash scripts/verify_submission.sh
bash scripts/build_audio_sentinel_app.sh
out/audio-sentinel/audiodetect input.wav --debug
```

`input.wav` 使用 16 kHz、PCM16、单声道格式。录屏可运行菜单，或分别执行五项演示：

```bash
bash demos/run_demo.sh      # 五项演示菜单
bash demos/classify_demo.sh # 五类识别
bash demos/robust_demo.sh   # 现场加噪与分类
bash demos/model_demo.sh    # C 模型规模与测速
bash demos/alarm_demo.sh    # HTTP/MQTT 本地回环
bash demos/keyword_demo.sh  # 合成关键词识别
```

[demos 说明](demos/README.md)包含 VS Code Remote SSH 连接和本地音频配置。演示先执行实际 Linux C 程序，已有实验表显示在运行结果末尾。Linux 主机 C、完整 OpenVela 系统模拟器与真实开发板分别验证；上述脚本运行于 Linux 主机 C。

## 工程与文档

|目录|内容|
|---|---|
|`app/audiodetect/`|CLI、LVGL 界面、共享 C 特征与推理、关键词、网络接口|
|`training/mlp_study/`|特征缓存、固定配方训练、基线、消融与导出|
|`model/`|正式部署模型和最近一轮 60 个实验模型|
|`board/`|R528S3 小屏配置、启动脚本及 LVGL 补丁|
|`results/`|Linux 实验、真板测量和逐条结果|
|`demos/`|Ubuntu Linux C 五项录屏演示|

[实验说明](docs/EVALUATION.md) · [实现说明](docs/mlp-primary.md) · [板级部署](board/README.md) · [训练复现](training/mlp_study/README.md) · [开源发布](docs/PUBLIC_RELEASE.md) · [第三方资源](THIRD_PARTY_NOTICES.md)

原始音频、私钥、虚拟机、厂商库和烧录镜像由使用者本地管理。项目为研究与演示原型，适用于声音提示及端侧技术验证。
