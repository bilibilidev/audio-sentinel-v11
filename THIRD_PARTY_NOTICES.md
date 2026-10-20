# 开源及第三方资源说明

本仓库仅发布Audio Sentinel v11的应用源码、最终模型、生成权重、板级配置和既有离线结果。不含原始数据集、私人日志、报告视频、完整OpenVela SDK、厂商预编译库或固件镜像。

根目录Apache-2.0许可适用于团队有权以该许可发布的源码，不能替代第三方软件、数据、模型、模板或素材的适用条款。公开可访问不等于可以任意再分发。以下状态为2026年10月10日公开整理时的核查结果，不是全部授权已完成的声明。

## 资源清单

| 资源与类型 | 版本与来源 | 使用和开放方式 | 许可与主要义务 | 自主开发边界与核查状态 |
|---|---|---|---|---|
| OpenVela / NuttX / nuttx-apps，系统软件 | `openvela.xml`引用`dev-ai-contest-2026`；历史实际构建SHA未完整锁定 | 调度、文件系统、采集与应用集成；库本体不在本仓库 | 核心采用Apache-2.0；单独第三方目录另核对；保留适用版权、LICENSE/NOTICE和修改说明 | 团队新增应用和配置，未开发内核与音频驱动；历史精确版本待核验 |
| LVGL，图形组件 | `apps_graphics_lvgl`同名分支，实际构建版本未锁定 | 本地界面；仅发布应用调用代码 | MIT；再分发库或实质部分时保留版权、许可和免责声明，附属字体另核对 | 团队实现页面，未开发LVGL；分支`LICENCE.txt`已核验，实际构建提交待核验 |
| ESC-50，音频数据集 | Karol J. Piczak / ESC-50；历史下载提交、压缩包哈希未保留 | 四类事件及其余类别背景，用于训练与五折；原始/增强音频和缓存不再分发 | 整体CC BY-NC 3.0；保留署名、来源和非商业限制，单片段依原始许可处理 | 团队开发数据和训练流程；整体许可已核验，历史所选片段逐项记录待补 |
| Python，解释器 | Python 3.10+；历史发行包未锁定 | 训练和导出；运行时不再分发 | PSF及发行包附带条款；再分发时核对并保留相应通知 | 团队编写脚本，未开发解释器；实际历史发行包待核验 |
| NumPy，计算库 | `>=1.24,<3`；历史安装版本未记录 | 训练、特征与导出；库不再分发 | 核心BSD-3-Clause；保留版权、条款和免责声明，不以作者名义背书，附属库另核对 | 团队实现训练流程；1.24.0核心许可作为参考已核验，不代表实装版本 |
| Allwinner / Rivotek BSP，板级与预编译组件 | `vendor_allwinnertech`分支引用，历史组件未锁定 | 显示、触摸、音频与启动；仅发布配置/rcS增量 | 依文件及厂商二进制条款分别处理；当前有厂商保留权利和二进制交付说明，不是全量Apache-2.0 | 团队调整配置与应用启动，未开发厂商驱动；历史链接库及固件分发权待核验 |
| C编译器、交叉工具链与构建工具 | 入口见脚本和manifest，历史发行版未锁定 | 编译；工具链和静态库不再分发 | 按实际发行包和运行库条款核对；发布固件前检查链接内容、通知和相关义务 | 团队编写构建/overlay入口；实际版本及产物分发条件待核验 |
| v11 TinyMLP及权重，团队训练产物 | NPZ哈希见`model/README.md`；无第三方预训练声学模型 | 提供最终NPZ及三份C权重供研究复现；不含每折模型 | 数据来源和权重边界见下文；源码许可不保证权重可无条件商用 | 团队训练、量化、导出和集成；文件已核验，商业用途与共同权利人授权另核对 |
| AI辅助开发工具，第三方服务 | 曾使用AI辅助开发；历史完整工具/模型版本清单未固化 | 编码、排错和复核；非运行时依赖，不再分发工具或私人会话 | 服务与工具条款分别适用，输出需人工审核、来源核查与验证 | 设备分类由TinyMLP完成，不调用生成式AI；历史工具条款和开发明细待发布方核验 |
| MFCC+DTW关键词模板，合成特征 | `training/keyword_detect.py`共振峰合成流程及`app/audiodetect/keyword_templates.h` | 命令行识别原型；不含真人录音，不调用第三方TTS服务 | 项目源码与生成特征一并发布；不据此声明真实语音性能 | 按发布方确认标注为合成语音模板；未接入实时UI，未做真实录音独立评测，不纳入五类指标 |

## ESC-50署名与数据边界

- 名称：ESC-50: Dataset for Environmental Sound Classification。
- 作者：Karol J. Piczak。
- 项目：[ESC-50](https://github.com/karolpiczak/ESC-50)。
- 许可与逐片段署名：[上游LICENSE](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE)。
- 整体许可：Creative Commons Attribution-NonCommercial 3.0 Unported（CC BY-NC 3.0）。ESC-10子集的许可不能直接替代本项目ESC-50选择集的许可。

复现实验应另行获取数据，并保留上游`LICENSE`、`meta/esc50.csv`及实际采用片段的作者、来源和许可记录。单片段条款不完全相同，不将整个数据集当作CC0或无条件商用素材。复制、混音、改编与再分发前核对实际条款。

该公开许可本身提供授权。符合署名、非商业等许可条件的使用，不要求再次向作者单独申请许可；超出许可范围的使用需要另行取得相应授权。参赛的具体使用方式和赛事要求仍需分别核对，不能把非商业许可写成无条件商用许可。

训练采用时间平移、音量缩放、时间拉伸、加性噪声和同训练折背景混合；这些处理不取消原素材义务。本仓库不打包演示音轨或数据片段。

## 模型与源码的区分

2026-10-17 新增独立 CNN v1 候选：源码位于 `training/cnn/` 和 `simulator/cnn/`，六个权重文件位于 `model/cnn-v1/`，仍仅由 ESC-50 选择集训练，无第三方预训练声学模型。PyTorch 2.14.1+cpu 仅用于本次 CPU 训练与 Python 参考，不随仓库再分发；C 推理无此依赖。C FFT 结构沿用本仓库 `app/audiodetect/keyword_detect.c` 并保留来源说明。原 v11 MLP 的范围和历史状态不变。

外部 FSD50K 录音不加入梯度训练、不随 Git 分发。逐条来源、许可链接及哈希保存在 `results/cnn-v1/external-reviewed-45.json`；该清单不改变原始录音条款。2026-10-19，用户确认另 20 条 Donate-a-Cry 婴儿录音标签正确，组成共 65 条的外部开发验证集。CNN v2 第四轮使用该集合选择候选，因而它不是未见过的盲测集；原始音频仍不随 Git 分发。模型用途、数据许可及商业使用的边界继续适用下文说明。

2026-10-19 的 CNN v2 当前候选沿用仅以 ESC-50 选择集训练的 CNN v1 权重进行微调。另行开展的 PANNs Cnn6 冻结特征基线和教师特征蒸馏实验使用了 AudioSet 预训练先验，应与仅使用 ESC-50 的实验分开比较。官方权重来自 [PANNs 官方发布记录](https://zenodo.org/records/3987831)，模型实现来源为 [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)，固定提交和权重哈希见 `results/cnn-v2/panns-provenance.json`。本仓库不再分发该第三方原始权重或上游模型源码；本说明不额外授予其许可。

模型和权重作为当前研究原型的一部分保留。训练来源与数据条款继续需要核对，不对商业使用作额外授权保证；商业部署或再分发前应确认实际素材、共同权利人和适用条款。本说明不宣称权重当然继承某一种数据许可，也不替代具体权利核查。合成语音关键词模板、DTW识别和生成脚本按v11快照保留，来源与性能边界见上表；生成脚本中的合成测试结果不能当作真实语音识别准确率。

## 外部系统与厂商内容

manifest只是外部资源获取入口，不表示资源被本项目重新许可。恢复板端或发布完整固件前应保存实际`repo manifest -r`、工具链版本、链接组件及通知，再决定可公开范围。本次未重建环境或发布厂商预编译库。

## 许可核验来源

- [OpenVela NuttX](https://github.com/open-vela/nuttx)、[nuttx-apps](https://github.com/open-vela/nuttx-apps)。
- [LVGL许可](https://github.com/open-vela/apps_graphics_lvgl/blob/dev-ai-contest-2026/LICENCE.txt)。
- [Python许可](https://docs.python.org/3/license.html)。
- [NumPy 1.24.0参考许可](https://github.com/numpy/numpy/blob/v1.24.0/LICENSE.txt)。
- [厂商BSP声明](https://github.com/open-vela/vendor_allwinnertech/blob/dev-ai-contest-2026/LICENSE)。

这些分支和参考版本是核查入口，不冒充历史构建依赖。未核验项应后续补证，不能无证据改成“已合规”。



## 2026-10-20 真板适配补充

本次发布 R528 小屏 LVGL 驱动的最小适配 patch，上游父提交为 `0f2a49f588505a00e8b46e25a34581c87291a62a`，验证提交为 `9465c48a115dcd6f00f643b7fadae81e1da2f9d1`。保留实际 SDK 的 MIT 许可原文于 `board/r528s3-gemini-s1/patches/LVGL-LICENSE.txt`；该补丁不将 LVGL 重新许可为项目根许可。新构建依赖以 `results/board-20261020/manifest-resolved.xml` 为准，不能据此反推旧 v11 的历史依赖。公开构建/评测日志已排除私人运行配置，不发布原始音频、预训练权重、厂商库或固件二进制。
