# 正式 Tiny MLP（模型2026-10-24，复测2026-10-26）

默认App现为92→96→5 Tiny MLP，9,413参数。最终模型从随机权重训练60轮，不加载v11/CNN权重，不微调，不使用教师。用户已选择结束CNN搜索；原CNN和旧MLP均保留。

|完整录音分类|正式MLP优化版|上一版重训MLP|历史部署CNN v2|
|---|---:|---:|---:|
|ESC五类官方保留折|337/390，86.41%，固定配方|318/390，81.54%|316/390，81.03%|
|外部65条|57/65，87.69%|54/65，83.08%|52/65，80.00%|
|外部婴儿20条|17/20，85%|18/20，90%|13/20，65%|

**口径不同**：2026-10-26新结果为同一92→96→5正式配方、390条、每折固定60epochs；宏召回84.24%，咳嗽召回70%。旧内层选型333/390（85.38%）含不同宽度/扩充背景，本轮Linux复测仍为333/390；该历史数据保留，不把不同协议的分数差异归因于权重升级。全量重新训练的参数与正式模型一致，未替换模型。

外部65条包含20条婴儿，未进入梯度训练，但已被多轮开发/选型使用；不是新盲测。婴儿比上一版少对一条，总体提高并不代表每类提高。以上为argmax完整录音分类，不应用0.80门限，不代表麦克风现场告警率。

## 训练与推理

- 92维原始特征保持：前40维时域统计做`asinh(x/scale)`压缩，其余52维MFCC/Delta不做该压缩，再做标准化。尺度、均值、标准差均只拟合对应训练集。
- 候选比较隐藏层64/96/128、温和增强、窗口监督/录音级监督、权重EMA等；正式全量模型选择`window96_asinh`，训练390条、60轮，AdamW，初始学习率0.002、weight decay0.001、label smoothing0.02、dropout0.15（仅训练）。
- 各折内层只用其余三折训练、下一折验证；重新随机初始化后在外层四折训练，再对保留折评分。扩充背景候选可用ESC其余1602条背景，额外8条跨折源录音已排除；最终全量正式模型未用额外背景。
- INT8仅为权重/偏置存储，特征、asinh及网络运算为浮点。全文件按一秒窗口概率均值聚合，尾窗补零，没有低能量强制背景和事件最大值覆盖。
- 文件读取分块保持结果一致；MLP仍是一秒窗口模型，UI仍采集三秒arecord文件。CNN连续因果状态和100ms EMA属于保留CNN入口，不作为MLP功能。

## 构建与切换

```sh
sh scripts/verify_submission.sh
sh scripts/build_audio_sentinel_app.sh
out/audio-sentinel/audiodetect input.wav --debug
out/audio-sentinel/audiodetect keyword keyword.wav
out/audio-sentinel/audiodetect input.wav --report

# 保留CNN回退，输出目录独立。
sh scripts/build_cnn_v2_app.sh
out/cnn-v2/audiodetect input.wav --debug
sh scripts/build_cnn_v2_stream.sh
out/cnn-v2/cnn_stream --live input.wav
```

普通主机需要现有C11编译器和数学库。OpenVela的`CONFIG_AE_AUDIODETECT_CNN`默认关闭，CLI/UI共用MLP；显式开启则使用保留CNN。`build_r528s3_cnn_image.sh`仍是专门的CNN镜像构建脚本，不能把它生成的镜像称为新版MLP固件。

## 验收和已知限制

正式App65条预测与独立C评测全部一致；70条输入的PCM/WAV接口一致、空PCM/非法WAV拒绝、5种长度数字静音均不触发0.80告警。静音原始argmax为玻璃（约0.464），未为静音样例额外强制改判。

关键词保留MFCC+DTW合成模板CLI原型，只读前一秒、未接入UI；合成回归测试不是真人语音准确率。网络保留显式`--report`到127.0.0.1的HTTP/MQTT原型；HTTP状态匹配/分段响应和MQTT CONNACK严格校验等已知问题仍在，无TLS、可靠队列或UI自动上报。

2026-10-24首次为Windows原生C；2026-10-26已在实际Ubuntu22.04/GCC11.4上完成正式模型、10组消融、PCM/WAV、读取块、ASan/UBSan、合成关键词/本机网络验收，见[新实验](../results/mlp-study-20261026/README.md)。三秒WAV函数均值2.482ms、P95 2.541ms，峰值CLI RSS2624KiB；均不是板端数据或声音至告警时延。

全片0.80门限下婴儿仅4/20正确触发；5dB高斯噪声五折为267/390（68.46%），宏召回48.41%。以上限制与分类准确率分别记录。本轮**仍未做完整OpenVela模拟器或开发板验收**，未重新打包/烧录镜像，不复用旧CNN的223.3ms板端时延。旧模型/导出头文件/CNN/镜像均保留。

完整逐条结果、各折配置和负面对照见[优化结果](../results/mlp-optimized-20261024/README.md)。
