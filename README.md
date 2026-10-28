# Audio Sentinel：OpenVela 离线声音事件检测

Audio Sentinel 在设备本地完成声音特征提取、Tiny MLP 分类与 LVGL 显示，识别背景、咳嗽、玻璃破碎、婴儿哭声和狗叫。正式模型为 **92→96→5，9,413个参数**，采用INT8权重/偏置存储和浮点运算，从随机初始化训练60轮。

## 功能与实现

16 kHz、16-bit、单声道音频经过92维特征提取，首40维时域统计做asinh压缩，后52维由MFCC及Delta组成。完整录音按一秒窗口处理，对全部窗口的类别概率取平均。CLI与LVGL界面共用分类核心；界面按三秒文件采集，提供START、STOP、AUTO和事件摘要。

命令行同时提供多SNR对照、MFCC+DTW关键词识别和HTTP/MQTT事件上报。关键词模板来自合成语音；网络演示采用127.0.0.1回环。当前项目目录只保留正式Tiny MLP及最近一轮实验所需代码，旧版本可通过Git历史查阅。

## 实验结果

| 模型 | ESC五折 | 外部开发集 | 婴儿哭声 |
|---|---|---|---|
| Tiny MLP | 337/390，86.41% | 57/65，87.69% | 17/20 |
| MFCC SVM | 312/390，80.00% | 39/65，60.00% | 12/20 |
| MFCC RF | 299/390，76.67% | 38/65，58.46% | 11/20 |
| PANNs Cnn6 linearSVM | 370/390，94.87% | 63/65，96.92% | 19/20 |

ESC结果采用五类390条录音的官方五折，正式MLP为固定配方评估。外部集包括FSD50K的45条录音和Donate-a-Cry的20条婴儿录音，已用于开发选型。PANNs使用额外AudioSet预训练，其精度更高；Tiny MLP的取舍是较小的存储和计算规模。

R528S3真板完整WAV分类为57/65，与Linux C的65条预测一致。三秒RAM WAV处理均值84.067 ms，包含读取与推理，录音时间另计。五次麦克风采集和200次短时循环已完成；NAND批量写入仍有SPI0/YAFFS故障，板端网络上报和长期稳定性尚待完善。功耗未实测。

## Linux C 与视频演示

在Ubuntu终端编译并推理：

```bash
bash scripts/verify_submission.sh
bash scripts/build_audio_sentinel_app.sh
out/audio-sentinel/audiodetect input.wav --debug
```

`input.wav`使用16 kHz、PCM16、单声道格式。五个亮点各有独立入口：

```bash
bash demos/classify_demo.sh  # 五类识别
bash demos/robust_demo.sh    # 现场加噪与分类
bash demos/model_demo.sh     # C模型规模与测速
bash demos/alarm_demo.sh     # HTTP/MQTT回环
bash demos/keyword_demo.sh   # 关键词识别
```

[demos说明](demos/README.md)包含VS Code Remote SSH连接和本地音频配置。音频示例实际执行Linux C，实验表读取已保存记录。Linux主机C、完整OpenVela系统模拟器和真实开发板分别描述，本次演示属于Linux主机C。

## 工程与文档

|目录|内容|
|---|---|
|`app/audiodetect/`|CLI、LVGL界面、共享C特征与推理、关键词、网络接口|
|`training/mlp_study/`|特征缓存、固定配方训练、基线、消融与导出|
|`model/`|正式部署模型和最近一轮60个实验模型|
|`board/`|R528S3小屏配置、启动脚本及LVGL补丁|
|`results/`|Linux实验、真板测量和逐条结果|
|`demos/`|Ubuntu Linux C五项录屏演示|

[实验说明](docs/EVALUATION.md) · [实现说明](docs/mlp-primary.md) · [板级部署](board/README.md) · [训练复现](training/mlp_study/README.md) · [开源发布](docs/PUBLIC_RELEASE.md) · [第三方资源](THIRD_PARTY_NOTICES.md)

原始音频、私钥、虚拟机、厂商库和烧录镜像由使用者本地管理。项目为研究与演示原型，适用于声音提示及端侧技术验证。
