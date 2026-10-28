# Audio Sentinel OpenVela 应用

应用包含CLI、LVGL界面、共享Tiny MLP分类器、WAV读取器、MFCC+DTW关键词和HTTP/MQTT告警接口。正式模型为92→96→5，使用INT8权重存储及浮点运算。

## CLI

```text
audiodetect input.wav --debug
audiodetect input.wav --report
audiodetect robust input.wav
audiodetect keyword input.wav
audiodetect ui
```

输入为16 kHz、PCM16、单声道WAV。完整录音以一秒窗口概率均值分类，末段补零；`robust`原型只处理首一秒。关键词模板来自合成语音，`--report`采用代码中配置的服务器地址，默认127.0.0.1。

## UI 与构建

LVGL界面提供START、STOP、AUTO、类别与置信度显示、能量、窗口数、告警计数及最近三条事件摘要。采集按三秒文件完成。OpenVela使用本目录CMake/Make入口，`CONFIG_AE_AUDIODETECT_UI`启用图形界面；Linux入口为`bash scripts/build_audio_sentinel_app.sh`。

NuttX调度、显示、音频驱动和存储影响整机行为。板端最新运行情况见[真板实验](../../results/board-mlp-20261027/README.md)。
