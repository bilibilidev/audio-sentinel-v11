# 2026-10-24 从零训练与 Linux C 实测

官方保留折：Float 314/390（80.51%）；INT8 存储 Python 及实际 Linux C 均为 312/390（80.00%）。
外部 65 条：最终模型 52/65（80.00%），婴儿哭声 17/20（85.00%）。它们是曾参与项目开发选型的真实录音，仍不属于新增盲测集。
同一批外部 65 条上，五个 ESC 折模型 C 成绩依次为 44、48、45、48、48；这是模型间评测，不是外部数据五折交叉验证。
新模型与旧微调版外部总分持平；婴儿 +4、咳嗽 -1、玻璃 -1、狗叫 -2。不能宣称所有类别更好。

`runtime-linux-c.json`：6 模型外部测试、390 条保留折预测、App/stream 对齐、分块/前缀/PCM stdin 检查、ASan/UBSan。
390 个原始 WAV 哈希已核验；与 ESC 特征缓存最大误差 1.91e-6。C/Python 最大概率差 1.23e-6 以内。
这是实际 Ubuntu Linux x86_64 共用 C 核心运行；本轮未重建完整 OpenVela 模拟器镜像，也未重测开发板。

`keywords.json`：合成样本接口/CLI 冒烟检查，不能作为真实语音准确率。CLI 只读第一秒；界面未集成。
`network.json`：HTTP/MQTT 127.0.0.1 回环端到端通过；MQTT 为协议测试服务，非部署 broker。
仍有 HTTP 状态码误判、分段状态行失败、MQTT 非法 CONNACK 长度未拒绝三项已复现问题；UI 和关键词均不自动触发网络告警。

复现入口：`training/cnn_v2/train_from_scratch.py`、`evaluate_from_scratch.py`、`prepare_scratch_runtime.py`。
训练先冻结全部六个模型再打开外部数据；固定配方评测，不称为独立嵌套选型。
模型、头文件和启动命令见 `model/cnn-v2-scratch/README.md`。
完整命令、编译日志、二进制与逐帧轨迹存于本地独立评测目录，原模型保持不变；本轮未推送远端。
