# OpenVela环境重建提示词

请根据公开仓库 https://github.com/bilibilidev/audio-sentinel-v11 的main分支，恢复Audio Sentinel v11的Python评测、主机C推理和完整OpenVela模拟器环境。先检查现有系统架构、内存、磁盘、Ubuntu虚拟机及残留工作区，优先复用，不删除已有环境。Ubuntu构建路径参考 https://github.com/open-vela/docs/blob/dev/zh-cn/quickstart/openvela_ubuntu_quick_start.md 。涉及软件安装、虚拟机资源调整、网络下载、付费资源或更换系统时，先列出方案并取得确认。

唯一应用代码基准是上述个人仓库main，不从原赛事仓库、旧提交包或其他模型版本覆盖源码。拉取后记录HEAD提交；最终模型为model/tiny_mlp_mfcc92_robust_simulator.npz，SHA256必须为c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e。关键词识别按当前源码保留，但模板来自合成语音，只作为命令行原型；不要将合成测试结果写成真实语音准确率。

按顺序完成：①执行scripts/verify_submission.sh；②复用scripts/build_robust_c_simulator.sh编译并运行16kHz、16-bit、单声道WAV的C推理；③用项目audio-sentinel-v11.xml获取完整OpenVela工程，先检查当前manifest、LFS和工具链是否完整，不擅自把dev-ai-contest-2026替换为最新dev分支；④为模拟器单独选择合适的goldfish目标配置，按需启用audiodetect、LVGL和音频组件；⑤先验证NSH和WAV推理，再验证界面与音频采集。R528S3真板defconfig不能直接用于goldfish，不执行板级overlay覆盖模拟器配置，不烧录或修改真实开发板。

如果应用需要模拟器适配，先列出差异，在独立分支/工作区做最小改动，不改模型、门限、特征算法或训练策略，不推送，待我审查。不要未经要求重新训练；若需重算准确率，应按README用ESC-50完整五折流程，不能对最终模型训练数据评分后称作交叉验证。Mac上主机C程序运行成功不等于完整OpenVela模拟器恢复成功。云服务器如用于完整模拟器，先检查图形、音频和虚拟化条件，不预设这些能力存在。

保存实际repo manifest -r、依赖及工具链版本、构建命令、日志、输出路径、WAV测试来源和校验值，区分主机C、OpenVela模拟器和真板结果。最终给出逐项验收表、启动命令、未完成项，以及是否能重新构建应用镜像；未取得历史依赖锁定信息时，不宣称与原镜像字节一致。先给出环境检查结果和安装计划，获得确认后再执行环境搭建。
