# Tiny MLP 设计与实现

正式分类器采用92维输入、96单元ReLU隐藏层和5类Softmax输出，共9,413个参数。模型从随机初始化训练，训练数据为ESC-50固定五类子集的390条录音。归一化均值、标准差及asinh尺度均由训练折拟合。

音频前端输出40维时域统计和52维MFCC/Delta统计。时域部分先进行asinh压缩，再与其余特征共同标准化。C端保存INT8权重与偏置，并用浮点scale恢复计算，DSP、激活和网络运算仍采用浮点。权重/偏置占9,413 B，归一化及scale另占912 B。

## 录音分类流程

CLI接受16 kHz、PCM16、单声道WAV。读取器逐块取样，分类器累积成一秒窗口，末尾不足一秒补零。每个窗口产生五类概率，完整录音以全部窗口概率的均值作为结果。PCM与WAV接口共用窗口和聚合逻辑，文件读取块大小不会改变分类结果。

LVGL界面使用同一完整录音接口。一次采集生成三秒WAV，随后分析并更新类别、置信度、能量、窗口数及事件计数。该流程属于分段采集；连续无间隙PCM采集是后续工作。

## 事件与关键词

非背景结果且置信度大于0.80时产生本地事件。玻璃破碎的严重度为critical，婴儿哭声和狗叫为warning，咳嗽为info。`--report`将事件JSON发送到配置的HTTP/MQTT地址，当前演示地址为127.0.0.1。

关键词路径使用13维MFCC和DTW匹配“救命、帮助、报警”。模板来自项目合成语音生成器，属于独立命令行原型；五类分类和关键词识别采用不同处理路径。

## 文件与构建

共享实现位于`app/audiodetect/audio_classifier.c`、`feature_extractor.c`与`mlp_model_infer.c`。正式模型位于`model/mlp-optimized-20261024/final_all_esc50.npz`，导出入口为`training/mlp_study/export_model.py`。模型和C权重哈希记录在同目录的`deployment.json`中。

Linux构建使用`bash scripts/build_audio_sentinel_app.sh`。OpenVela由应用的CMake/Make入口编译相同核心，LVGL界面由`CONFIG_AE_AUDIODETECT_UI`启用。当前源码只提供正式Tiny MLP后端。
