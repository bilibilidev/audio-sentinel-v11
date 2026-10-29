# 开源发布说明

当前发布以Tiny MLP为唯一应用后端，包含共用C实现、LVGL界面、关键词原型、HTTP/MQTT接口、固定配方训练、标准基线、控制变量消融和Linux C演示。旧模型及旧实现从当前目录移除，版本演进由Git提交历史保存。

## 资源组织

正式部署模型位于`model/mlp-optimized-20261024/`，最近一轮实验模型位于`model/mlp-study-20261026/`。`results/mlp-study-20261026/`记录训练协议、基线、消融、噪声和Linux验证；`results/board-mlp-20261027/`记录真板分类、时延、采集和故障过程。

实验JSON、CSV及必要构建日志保留测量时的结果和版本信息。源码整理与说明文档更新不改变模型参数或已有测量值，当前脚本路径以仓库文档为准。

## 发布内容

仓库发布应用源码、权重、模型导出、板级配置及可公开实验数据。原始音频、PANNs原始checkpoint、厂商预编译库、完整SDK、固件镜像、虚拟机和个人凭据由本地工作区管理。提交用PDF、DOCX和视频由团队另行提供。

模型训练和评测数据分别遵循其原始许可。项目源码许可与第三方资源说明见[LICENSE](../LICENSE)、[NOTICE](../NOTICE)和[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。
