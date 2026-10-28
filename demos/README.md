# Linux C 五项实时演示

在 VS Code 的 Ubuntu Remote SSH 项目终端运行。五个入口都先执行当前程序，末尾再显示相应的实验记录。

```bash
cd ~/projects/audio-sentinel-v11-linux-demo
bash demos/classify_demo.sh   # 五条录音逐条分类
bash demos/robust_demo.sh     # 现场加噪并分类
bash demos/model_demo.sh      # C数组规模、RSS和WAV测速
bash demos/alarm_demo.sh      # HTTP/MQTT实际收发
bash demos/keyword_demo.sh    # 合成关键词9正例、2负例
```

每条命令只运行一项，按Enter继续。`bash demos/run_demo.sh`打开菜单；`bash demos/run_demo.sh --section all --auto`连续运行五项，并把所有实验记录统一放在末尾。

## 本次运行的内容

|亮点|现场操作|
|---|---|
|五类识别|从本地录音库读取背景、咳嗽、玻璃、婴儿哭声和狗叫各一条，调用Linux C，打印实际分值和处理时间|
|噪声鲁棒性|保留完整录音，按RMS现场生成20/10/5 dB高斯噪声WAV，运行20次C分类，显示五条示例的分类变化|
|模型轻量化|编译`model_benchmark.c`，用C的sizeof读取参数和存储字节，处理三秒WAV并测量均值、P95及进程峰值RSS|
|网络告警|启动127.0.0.1 HTTP和MQTT接收端，先运行背景，再运行咳嗽事件，展示实际收到的JSON|
|关键词识别|调用C的MFCC+DTW，运行三组“救命、帮助、报警”合成语音及静音、低噪声负例，显示本次结果|

示例采用本地已有的固定录音，音频推理调用`app/audiodetect/`共用C核心。Python负责输入准备、表格和接收端。模型、特征和告警门限保持当前正式配置。

轻量化测速先预热3次，默认测20次：

```bash
bash demos/model_demo.sh --repeats 30
```

计时包含WAV读取、特征提取和MLP处理。ELF大小、权重数组和RSS分别显示；RSS取自Linux当前程序的`/proc/self/status`，峰值对应`VmHWM`。C程序按实际数组读取INT8存储规模，并按当前窗口数核算网络MAC，FP32存储核算及PANNs对照放在末尾的实验记录中。

## 实验记录

每段现场结果之后，单独列出`results/mlp-study-20261026/`中的已有五折、噪声、轻量化、消融或网络/关键词记录。录屏时先展示操作和当前结果，再按Enter翻到记录表。五条声音示例、合成关键词与390条五折数据分别标注。

## 输入、编译和结果文件

当前Ubuntu已配置录音目录。其他机器复制`local_paths.example.json`为`local_paths.json`，填写Linux音频和关键词目录。需要已有Python 3.9+与GCC，脚本使用Python标准库。

首次运行自动编译Linux C。也可手动执行：

```bash
bash demos/build_demo.sh
```

产物位于`out/linux-demo/`，本次结果位于：

```text
out/linux-demo/live/latest-classify.json
out/linux-demo/live/latest-robust.json
out/linux-demo/live/latest-model.json
out/linux-demo/live/latest-alarm.json
out/linux-demo/live/latest-keyword.json
```

每次运行另建带时间和进程号的子目录，保存原始C输出、实际分值、计时和加噪输入。上述目录及原始音频不上传Git。

可选`--play`使用Ubuntu已有的aplay播放原始示例；声音由来宾音频设备输出。默认只运行终端。

## VS Code 连接

按Ctrl+Shift+P选择 **Remote-SSH: Connect to Host...**，连接本机配置的`audio-sentinel-linux`，打开`~/projects/audio-sentinel-v11-linux-demo`。左下角显示SSH主机名后新建终端。[官方连接说明](https://code.visualstudio.com/docs/remote/ssh)

终端五段分别录制后，再接真实开发板操作视频。演示运行在Linux主机C环境，真板操作另行展示。建议使用18–22号终端字体，隐藏个人路径、关闭通知；导出后检查MP4时长及文件大小。
