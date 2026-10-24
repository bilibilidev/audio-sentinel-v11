# tiny MLP v11 外部65条重测（2026-10-24）

模型保持为 `c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e`，92→64→5、6,277参数、INT8权重存储/浮点运算。三份C权重与NPZ在内存重新导出内容完全一致；未写入、未训练。

同一批65条16kHz单声道PCM16输入，逐文件哈希与CNN评测一致。实际运行平台为Windows原生C、MinGW GCC6.3.0；Ubuntu当前连接中断，本轮不是Linux或开发板实测。

|口径|总正确|婴儿正确|
|---|---:|---:|
|旧App原样：每条前3秒|53/65（81.54%）|18/20（90%）|
|评测适配：完整录音|52/65（80.00%）|17/20（85%）|
|原微调CNN v2，完整录音|52/65（80.00%）|13/20（65%）|
|从零训练CNN v2，完整录音|52/65（80.00%）|17/20（85%）|

原App结果经仓库原 `board_simulator_harness.c` 再核对，65/65预测与分值一致。完整录音模式仅扩展读取范围，复用原一秒特征、静音门限2e-5、非背景分值>=0.80优先取最大，否则均值聚合；末窗补零。该模式为独立评测适配，不是旧App原本支持完整录音。

本次不重新测试五折。65条仍为已见开发验证集，不能据1条差距断言某架构普遍更优。
与此前旧固件真板38/65的记录应分开：该记录模型哈希未能确认；不能将本轮同名保留模型的主机成绩当成旧固件复测。

源码：`scripts/evaluate_legacy_mlp.py`、`scripts/legacy_mlp_evaluation.c`。Linux现有GCC可用以下入口；本轮实际命令/编译日志保留在本地独立目录：

```sh
python3 scripts/evaluate_legacy_mlp.py --prepare /path/to/frozen-external-evaluation --output /path/to/private-mlp-run --run
```

未修改原模型、App默认CNN权重或开发板固件；未推送GitHub。
