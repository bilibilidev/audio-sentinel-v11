# 默认 Tiny MLP（2026-10-24）

当前 App 默认使用从随机权重重新训练的 92→64→5 MLP，6,277 参数。五个保留折和一个全量模型分别训练120轮，不加载旧权重、不微调、不使用教师；外部65条录音仅在12个MLP/CNN模型联合冻结后评分，本轮未按外部分数选择种子或epoch。

|完整录音分类|新默认MLP|同次重训CNN|原部署CNN|
|---|---:|---:|---:|
|ESC官方五折|318/390，81.54%|317/390，81.28%|316/390，81.03%|
|外部65条|54/65，83.08%|49/65，75.38%|52/65，80.00%|
|外部婴儿20条|18/20，90%|11/20，55%|13/20，65%|

MLP是本项目优先外部/婴儿场景的主版本，并非全类别胜出：五折宏召回78.04%低于同次CNN的81.26%；外部咳嗽、玻璃、狗叫也较低。外部65条属于已见开发集，不是新盲测；分类率未应用0.80告警门限。

本次默认后端选择也参考了这65条成绩。训练阶段未使用外部录音、未选种子/epoch，与训练完成后参考外部成绩选择MLP作为默认，是两个阶段；完整决策见`results/retraining-audit-20261024/selection-decision.json`。

## 构建和切换

```sh
sh scripts/verify_submission.sh
sh scripts/build_audio_sentinel_app.sh
out/audio-sentinel/audiodetect input.wav --debug
out/audio-sentinel/audiodetect keyword keyword.wav
out/audio-sentinel/audiodetect input.wav --report

# 保留原CNN回退，独立输出目录。
sh scripts/build_cnn_v2_app.sh
out/cnn-v2/audiodetect input.wav --debug
sh scripts/build_cnn_v2_stream.sh
out/cnn-v2/cnn_stream --live input.wav
```

普通主机需要现有C11编译器及数学库。OpenVela的`CONFIG_AE_AUDIODETECT_CNN`默认关闭，CLI/UI使用MLP；开启则使用保留的CNN。专用`build_r528s3_cnn_image.sh`显式开启CNN，不会因默认变化悄悄打包MLP。

## 推理规则

MLP每秒计算92维时域/MFCC/Delta特征及五类概率，完整文件取全部窗口概率平均；末尾不足一秒补零。预处理、归一化及权重与评测一致，不再使用旧的低能量强制背景或高置信度最大事件覆盖。告警仍要求非背景且置信度约0.80，CLI使用`>0.80`，UI使用`>=0.80`。

INT8仅为权重/偏置存储，运算为浮点；归一化每窗计算一次，避免每个隐藏单元重复计算。文件读取分块不改变结果，但MLP仍按一秒窗口分类，UI仍按三秒arecord文件采集。连续因果状态、100ms EMA更新属于保留的CNN流式入口，不能作为MLP功能描述。

数字全零输入的原始argmax可能是玻璃（约0.422），低于告警阈值，验证中未触发告警；没有为了静音样例再强制改判。低置信度/未知输入及现场误报率仍需完善。

## 保留功能与验收边界

- 关键词为MFCC+DTW合成模板原型，命令行仍只读前一秒，未接入UI，不宣称真人识别准确率。
- HTTP/MQTT为显式`--report`的127.0.0.1本机回环。新默认已实测发送，但HTTP状态解析/分段响应、MQTT CONNACK严格校验等已知缺陷仍保留；没有TLS、可靠队列或UI自动上报。
- 本轮训练和默认App验收在Windows CPU/原生C完成，完整OpenVela模拟器和真实开发板的新版本未验收；不复用旧CNN的223.3ms板端时延或功耗结论。
- 已烧录镜像、原v11 MLP权重和CNN模型全部保留；本次未重新打包或烧录开发板。

详细训练审计、逐条结果和完整矩阵见[`results/retraining-audit-20261024`](../results/retraining-audit-20261024/README.md)。
