# R528S3 Tiny MLP 真板实验

实验平台为R528S3 Gemini S1、OpenVela及板载2.8寸ILI9341小屏。烧录镜像使用正式Tiny MLP权重，模型哈希为`2c74151c…61610f9e9`，镜像源码提交为`72f4dfde5224a4e590ff66f4ce1791f5193e7b8c`。

## 完整录音分类

外部65条WAV在板端得到57/65（87.69%），与Linux C的65条预测一致，最大概率绝对差约2.1532e-5。五类统一计入上述65条结果，逐类统计保留在`clean65.json`。

外部65条来自FSD50K与Donate-a-Cry，标签已听审，已用于开发选型。五类结果按完整录音argmax计算，0.80事件门限后的触发情况在逐条记录中单独统计。现场麦克风分类率与每小时误报需要带标签的独立采集。

## 处理时间

|输入与存储|均值|P95|范围|
|---|---:|---:|---|
|固定三秒RAM WAV|84.067 ms|85 ms|读取、DSP、推理及命令输出|
|65条变长NAND WAV|212.692 ms|491.6 ms|录音长短不同，逐条CLI处理|

计时不含录音、USB传输和声音出现至界面告警的等待时间。固定输入经过预热后测量，原始分布见`latency.json`。板端功率与单次能耗尚无仪器测量结果。

## 采集与运行

小屏、触摸、自启及STOP操作已有运行记录。五次三秒采集均生成16 kHz、PCM16、单声道48000帧WAV，使用RAM文件系统；这些录音用于采集流程检查，没有事件真值标签。200次推理约持续69秒，属于短时循环实验。

批量NAND写入时出现SPI0超时及YAFFS重试，后续输入和采集改用RAM。板端HTTP/MQTT未送达，设备时间仍为1970年。当前系统工作集中在NAND写入、网络配置、连续采集和长期稳定性。

## 结果文件

`clean65.json`、`predictions.csv`记录逐条分类，`input-manifest.json`记录输入哈希，`latency.json`、`capture.json`和`soak.json`分别记录时延、采集及短时循环。`failed-attempts/`与`logs/`保存故障过程，`source-evidence/`保存本轮板测程序，`manifest-resolved.xml`及`build/`记录该镜像依赖与构建。

`python3 scripts/board_benchmark_mlp/verify_results.py`在仓库根目录核对结果计算及文件哈希，无需连接开发板。该结果对应上述已测试镜像；当前源码整理后新增构建应单独记录新产物版本。
