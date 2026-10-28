# OpenVela 应用运行说明

Audio Sentinel 将Tiny MLP、音频前端、关键词识别和事件上报封装为NuttX应用。CLI与LVGL调用同一分类接口，模型输入为16 kHz、16-bit、单声道音频。

## 当前实现

CLI提供完整WAV分类、单条首秒多SNR对照、MFCC+DTW关键词及可选HTTP/MQTT上报。LVGL提供START、STOP、AUTO、运行状态、类别、置信度、能量、处理窗口数、告警计数和最近三条内存事件摘要。自动启动偏好保存到本地配置。

UI一次采集三秒文件，再进行完整录音分析。模型内部仍按一秒窗口取概率均值，文件分块读取只影响输入方式。后续可在采集缓冲与推理之间建立连续PCM通道，减少录音段之间的间隔。

## 平台进展

|平台|当前结果|
|---|---|
|Ubuntu原生C|五折、外部WAV、接口一致性、异常输入和本机告警完成|
|OpenVela完整goldfish模拟器|当前Tiny MLP版本尚待完整系统复验|
|R528S3 Gemini S1|小屏、触摸、自启、WAV分类和RAM采集已有实验记录|

真板NAND批量写入仍存在SPI0/YAFFS问题；网络上报及长期连续运行需要继续完善。五次麦克风采集用于验证文件格式和采集流程，现场分类性能需要带事件标签的录音测试。

## 启动

在NuttX终端运行 `audiodetect ui` 打开界面，或运行 `audiodetect /data/input.wav --debug` 分类文件。Linux主机可通过[demos](../demos/README.md)展示共用C核心。R528显示与内存配置用于Gemini S1小屏平台；goldfish使用其自身的显示、音频和内存配置。
