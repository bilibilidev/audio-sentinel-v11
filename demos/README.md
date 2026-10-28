# Linux C 五项演示

演示运行在Ubuntu中，音频推理调用与OpenVela App共用的C核心，编译产物为Linux ELF。Python负责菜单、表格排版和本机测试接收端。

## VS Code 连接

在Windows端安装好的Remote SSH中，按Ctrl+Shift+P选择 **Remote-SSH: Connect to Host...**，连接本机已配置的 `audio-sentinel-linux`。进入Linux远程窗口后，打开 `~/projects/audio-sentinel-v11-linux-demo`，左下角显示SSH主机名时再新建终端。首次连接由VS Code配置远程Server。[官方说明](https://code.visualstudio.com/docs/remote/ssh)

不同工作机可使用自己的SSH主机别名及Linux源码目录。原始音频和密钥路径留在本地配置中。

## 编译与独立入口

在Ubuntu远程终端运行：

```bash
cd ~/projects/audio-sentinel-v11-linux-demo
bash demos/build_demo.sh
```

|亮点|命令|演示内容|
|---|---|---|
|1 五类识别|`bash demos/classify_demo.sh`|五个WAV示例的Linux C输出及已发布五折表|
|2 噪声鲁棒性|`bash demos/robust_demo.sh`|干净、20/10/5 dB五折对比表|
|3 模型轻量化|`bash demos/model_demo.sh`|PANNs对照、INT8存储、asinh消融和Linux耗时表|
|4 网络告警|`bash demos/alarm_demo.sh`|背景负例和咳嗽事件的HTTP/MQTT实际收发|
|5 关键词|`bash demos/keyword_demo.sh`|合成“救命、帮助、报警”和静音负例|

每条命令只运行一项，按Enter分段继续。`bash demos/run_demo.sh`提供选择菜单；`bash demos/run_demo.sh --section all --auto`用于快速检查五项。

编译使用 `scripts/build_audio_sentinel_app.sh`，产物为 `out/linux-demo/audiodetect`。同目录的`build.log`与`build.json`保存构建日志、工具链和哈希。正式模型、C头文件及输入WAV按SHA256校验。

## 实验表与实时输出

五类示例实际执行Linux C，并打印本次进程墙钟时间。噪声、轻量化和五折表读取`results/mlp-study-20261026/`的既有实验记录，不在录屏时重新训练或重跑五折。噪声可附加一条实时原型：

```bash
bash demos/robust_demo.sh --probe
```

原型只取WAV首一秒，和完整五折表分别展示。轻量化表比较9,413参数的Tiny MLP与4,837,455参数的PANNs，权重/偏置INT8存储为9,413 B，相对FP32核算减少75%。PANNs使用额外预训练且精度更高，功耗需要独立仪器测量。

## 告警和关键词

告警接收端只绑定Ubuntu内的127.0.0.1:8080/1883。Tiny MLP结果超过0.80门限后，C程序发送HTTP POST与MQTT PUBLISH，终端打印实际载荷并核对一致。退出时关闭自己的接收端，端口被占用时提示停止。MQTT测试端实现本演示的CONNECT和QoS 0 PUBLISH流程。

关键词采用MFCC+DTW，本次演示三条合成正例和静音；完整合成回归共9正例、2负例。真人语音性能与真板联网分别通过独立测试评价。关键词与网络告警是两条功能路径。

## 输入配置与录屏

Ubuntu需要现有Python 3.9+与GCC。复制 `local_paths.example.json` 为忽略的 `local_paths.json`，填写Linux音频目录和编译器。当前工作机已复用现有录音，脚本仅用Python标准库。

可选 `--play`调用来宾已有的aplay，声音通过Ubuntu音频设备输出；SSH本身不转发音频。默认只显示终端，不依赖音频设备。

五段分别录制，再接真实开发板操作视频。终端建议90–120秒，配两分钟真板片段约3.5–4分钟。终端字号18–22，隐藏个人路径、关闭通知；MP4 H.264/AAC 30fps、约4Mbps，五分钟约155MB，导出后核对实际大小。

当前演示属于Linux主机C；完整OpenVela模拟器和真实开发板分别呈现。外部65条是开发集，真板WAV时间统计另见结果目录。视频画面、字幕与旁白的身份信息在录制后统一检查。
