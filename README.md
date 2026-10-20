# Audio Sentinel：基于 OpenVela 的离线音频事件检测系统

当前开发版本：**CNN v2**；App 的 WAV 命令行和 LVGL 采集路径共用因果 CNN 核心。原 v11 MLP、CNN v1 及其模型与结果作为回退和对照保留。仓库不包含原始音频数据集、私人环境或构建产物。

### 真板验收与资源对照更新（2026-10-20）

分区擦除烧录后的 R528S3 Gemini S1 真板已完成 CNN v2 的 65 条外部 WAV 测试：**52/65（80.0%）**，婴儿 **13/20（65%）**，65 条预测与 Linux C 全部一致。固定三秒 WAV 推理均值 **223.3 ms**、P95 **224 ms**。公开逐条预测、输入哈希、构建来源、延迟原始值和复测脚本，见 [板级统计与对照表](results/board-20261020/README.md)。

PANNs Cnn6 冻结特征+SVM 外部为 **58/65（89.23%）**，精度更高；当前 CNN 仅 **14,885 参数**，约为其可学习网络参数的 **1/325**，5 秒网络 Conv/Linear MAC 约为 **1/674**。主机单线程含前端耗时的对照另列，PANNs 尚未在板上部署。没有功率计，**功率和能耗未实测**，不把计算量直接当作功耗。外部集参与过开发选型，以上不是盲测或实际麦克风准确率。

本次还同步已验证的 128 MiB DDR/RAMLOG 边界、小屏 ILI9341 注册分支、LVGL 对齐缓冲补丁及 UI 自启配置，未改模型权重、特征和门限。小屏/自启已恢复，但 STOP 后残留录音和后续采集失败仍需解决，实际录音到告警时延、软重启稳定性与完整 goldfish v2 运行尚未验收。

### CNN v2 更新（2026-10-19）

当前部署候选为 `clean_anchor`。最终 Ubuntu C 与共用 App CLI 已在 65 条外部录音上通过一致性验收，正确 52/65（80.0%）、婴儿 13/20（65.0%）；v1 为 52/65、12/20，仅多识别一条婴儿录音。按现有 0.80 全片置信度阈值计数两版均为 7/20，不能宣称实时告警已有提升。该候选用外部开发集选型。2026-10-19 的 R528S3 / ILI9341 镜像曾完成编译打包；最新 2026-10-20 真板镜像及结果以上方更新为准。完整 goldfish 模拟器的 v2 运行尚未验收。

新增按 ESC-50 官方折进行的内层选型、标准 MFCC+SVM/RF 和 PANNs 冻结特征+SVM 对照，以及 time-only、MFCC-only、无增强、通道和 dilation 消融。外部 65 条录音均由用户确认标签，未用于梯度训练；它们参与过开发选型，不是全新盲测集。协议、实际运行验收和镜像哈希位于 `results/cnn-v2/`，训练入口见 [training/cnn_v2](training/cnn_v2/README.md)。

新默认入口为 `scripts/build_cnn_v2_app.sh` 和 `scripts/build_cnn_v2_stream.sh`。App 不再调用 MLP 的静音强制改判和一秒窗口投票，完整 WAV 的聚合与 Python 模型一致；本地告警置信度仍为 0.80。UI 保留三秒 `arecord` 文件采集，每个文件内部连续推理，尚不是无间断 PCM 采集。使用方式、模型版本和执行层级见 [CNN v2](docs/cnn-v2.md)。

### 历史首版更新：CNN v1（2026-10-17）

v1 新增 14,885 参数的因果 CNN，保持连续音频状态，每 100 ms 输出一次在线分值。历史入口 `./scripts/build_cnn_c_simulator.sh` 仍可构建，见 [CNN v1](docs/cnn-v1.md)。v1 发布时默认 OpenVela 应用使用原 MLP。

在同一组人工确认的 45 条外部录音上，实际 Linux C 的五个折模型平均准确率由 72.4% 提至 79.1%，单个最终模型由 77.8% 提至 88.9%（40/45）。ESC-50 官方保留折 INT8 准确率由 82.6% 降至 80.3%，因此该候选尚未全面优于原模型。外部集合已用于开发验证，只有四种参考类别；20 条婴儿哭声尚待复核。整片分类率不代表实时报警准确率，也不代表完整 OpenVela 模拟器或开发板的性能。

Audio Sentinel 面向家庭声音提示和隐私敏感场景，在 OpenVela 设备本地完成麦克风采集、音频特征提取、轻量模型推理和可视化提示。系统识别 `background`、`cough`、`glass_break`、`baby_cry`、`dog_bark` 五类声音；默认推理不依赖云端，也不上传原始音频。当前为研究与演示原型，不替代医疗诊断或安全认证报警设备。

下文保留原 v11 MLP 应用的架构和历史指标。新版默认 App、当前模型和验收以顶部 CNN v2 说明及其结果目录为准。关键词原型和板级配置仍予以保留。

## 一、作品亮点

- **真实数据训练**：使用 ESC-50 中四类目标事件，并从其余类别抽取背景样本。
- **严格五折验证**：按 ESC-50 官方折划分，同一源录音切出的窗口不会跨训练集和测试集。
- **轻量端侧模型**：92 维时域/MFCC/Delta 特征，TinyMLP `92→64→5`，共 6,277 个参数。
- **INT8 权重存储**：估算从 25,844 字节降至 7,029 字节，缩减 72.80%。
- **OpenVela 真板集成**：已接入 R528S3 音频、LCD、触摸、LVGL 和开机自启动。
- **本地交互**：实时显示状态、类别、置信度、能量、处理窗口数和告警计数。
- **鲁棒性策略**：训练阶段使用混合增强（时间平移、音量缩放、时间拉伸、高斯噪声，并混合同训练折真实背景）；设备端增加静音门限和 0.80 事件阈值。
- **强噪声鲁棒性**：10 dB / 5 dB 下宏平均召回率为 80.57% / 74.37%，强噪声下五类事件均可检出。
- **结果随源码提交**：训练脚本、最终模型、混淆矩阵、各类别指标与 C 推理验证入口均位于仓库内。

## 二、系统架构

```text
Python 训练端
ESC-50 → 1 s 分窗 → 92维特征 → 五折训练/验证 → NPZ模型 → Float/INT8 C头文件
                                                        │
                                                        ▼
OpenVela R528S3 端
麦克风 → arecord 3 s → 三个1 s窗口 → C特征提取 → INT8 TinyMLP
                                                      │
                                   静音门限 + 0.80事件阈值
                                                      │
                                                      ▼
                              LVGL界面 / 本地事件计数 / 最近告警
```

三个运行端严格区分：

| 运行端 | 职责 | 本仓库中的位置 |
|---|---|---|
| Python 训练与评测 | 数据准备、五折训练、噪声鲁棒性、模型导出 | `training/` |
| Linux 主机 C 模拟器 | 编译 C 特征、三秒决策逻辑和 INT8 权重 | `simulator/`、`scripts/build_robust_c_simulator.sh` |
| 真实 OpenVela 开发板 | 麦克风、LCD、触摸、LVGL 和自动启动 | `app/audiodetect/`、`board/` |

主机 C 模拟器与开发板共用同一套 92 维 C 特征实现。模型指标来自 ESC-50 离线五折评测，各折使用该折训练出的模型；部署模型为最终训练后导出的 NPZ，不能直接在其训练数据上评分并称作交叉验证。此段为历史 MLP 架构说明；当前 CNN 真板准确率、时延和内存见顶部 2026-10-20 结果，功耗仍未实测。

## 三、仓库结构

```text
audio-sentinel-v11/
├── app/audiodetect/                 # OpenVela/NuttX 应用源码与权重头文件
├── board/r528s3-gemini-s1/          # 当前真板 defconfig、rcS 和冻结配置
├── training/                        # 当前 Python 数据、训练、评测与导出源码
├── simulator/                        # 主机 C 模拟器 INT8 推理与生成权重
├── model/                            # 项目模型 NPZ
├── results/
│   └── python_cv/                    # 五折指标、混淆矩阵与噪声鲁棒性结果
├── docs/                             # 当前实验口径、功能边界与公开整理说明
├── scripts/                          # 板级 overlay、C 模拟器与仓库自检脚本
├── THIRD_PARTY_NOTICES.md            # 来源、许可义务与核验边界
├── NOTICE                            # 公开源码来源说明
└── audio-sentinel-v11.xml            # repo manifest，映射应用到 apps/audiodetect
```

### 快速开始：复用现有主机 C 推理入口

不需要重训、下载 ESC-50 或安装完整 OpenVela SDK。仓库已包含 v11 生成权重。

```bash
git clone https://github.com/bilibilidev/audio-sentinel-v11.git
cd audio-sentinel-v11
./scripts/verify_submission.sh
./scripts/build_robust_c_simulator.sh
./out/robust-simulator/audiodetect_robust_sim /path/to/16k_mono_16bit.wav
```

自检需要 Python 3；编译入口原目标为 Linux 主机，需要 C11 编译器和数学库。WAV由使用者提供，格式为16 kHz、16-bit单声道，建议使用3秒录音。上述入口沿用原 v11 MLP；新增 CNN 使用独立的构建命令和权重。自检不等于编译或真机验收。

## 四、拉取完整 OpenVela 工程

```bash
repo init -u https://github.com/bilibilidev/audio-sentinel-v11 \
  -b main -m audio-sentinel-v11.xml
repo sync -c -j8
```

同步完成后，本仓库位于工作区的 `audio-sentinel-v11/`，应用通过 manifest 映射到外层 `apps/audiodetect/`。OpenVela上游仍按现有manifest引用分支，未锁定历史构建的全部SHA；原完整环境和固件不随仓库提供，不保证与原镜像字节一致。厂商组件应按其获取与分发条款另行核对。

可先执行仓库完整性检查：

```bash
cd audio-sentinel-v11
./scripts/verify_submission.sh
```

## 五、Python 训练与交叉验证

### 5.1 环境

```bash
cd audio-sentinel-v11/training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 5.2 准备 ESC-50

将 ESC-50 放到 `training/data/datasets/ESC-50/`，目录中应包含 `audio/` 和 `meta/esc50.csv`。数据集本身不提交到本仓库。

```bash
python import_esc50.py \
  --esc50_dir data/datasets/ESC-50 \
  --output_dir data/raw_esc50_real \
  --background_per_class 5 --seed 42 --overwrite

python prepare_data.py \
  --raw_dir data/raw_esc50_real \
  --output_dir data/processed_esc50_real --overwrite
```

### 5.3 构建当前 92 维特征与增强缓存

```bash
python extract_features_mfcc.py \
  --labels data/processed_esc50_real/labels.csv \
  --out results/esc50_mfcc92_features.npz --include-delta

python build_robust_augmented_cache.py \
  --base-cache results/esc50_mfcc92_features.npz \
  --feature-mode mfcc92 --profile hybrid --copies 4 --seed 2026 \
  --out results/esc50_mfcc92_robust_augmented_features.npz
```

### 5.4 重新训练和五折验证

```bash
python train_edge_mfcc.py \
  --clean-cache results/esc50_mfcc92_features.npz \
  --aug-cache results/esc50_mfcc92_robust_augmented_features.npz \
  --models-dir models/edge_mfcc_cv \
  --final-model models/tiny_mlp_mfcc92_robust_simulator.npz \
  --report-dir results/python_cv \
  --epochs 120 --hidden-dim 64 --class-weight-power 0.5 \
  --aug-fraction 0.5 --event-aug-fraction 0.75 --background-aug-fraction 0.25 \
  --batch-size 64 --lr 0.003 --weight-decay 0.0001 --seed 42
```

模型 SHA256：

```text
c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e
```

## 六、Linux 主机 C 模拟器

主机 C 模拟器复用 `app/audiodetect/` 中的 C 特征提取与三秒决策逻辑，配合 `simulator/` 的 INT8 推理实现，编译出可独立运行的验证程序。

```bash
python training/export_simulator_int8.py \
  --model model/tiny_mlp_mfcc92_robust_simulator.npz

./scripts/build_robust_c_simulator.sh
./out/robust-simulator/audiodetect_robust_sim /path/to/16k_mono_16bit.wav
```

已提交生成权重时，可以直接编译而不运行导出命令。该入口用于确认 C 推理链路和门限行为；其样本不属于独立测试集，模型指标以 `results/python_cv/metrics.json` 为准。

## 七、真实 OpenVela 开发板构建

目标硬件为润芯微 OpenVela 适配开发板，构建配置对应 R528S3 Gemini S1。

在 `repo sync` 后的 OpenVela 工作区根目录执行：

```bash
./audio-sentinel-v11/scripts/apply_board_overlay.sh .

./nuttx/tools/configure.sh -e \
  vendor/allwinnertech/boards/r528/r528s3-gemini-s1/configs/nsh

make -C nuttx EXTRAFLAGS="-Wno-cpp -Wno-deprecated-declarations" -j16
```

板级源码和配置保持v11冻结状态。本仓库不提供完整SDK、烧录包或原固件；上述命令为原构建流程的入口，不是本次在恢复环境中重新构建的结果。最终烧录包还需按厂商流程打包。

### 7.1 运行方式

启用 `CONFIG_AE_AUDIODETECT_UI_AUTOSTART` 后，板级 `rcS.nsh` 会在硬件服务初始化后等待 3 秒并执行：

```text
audiodetect ui &
```

也可以从 NSH 手动运行：

```text
nsh> audiodetect ui
nsh> audiodetect /data/test.wav --int8 --debug
```

## 八、模型设计

### 8.1 特征

| 特征组 | 维度 | 内容 |
|---|---:|---|
| 时域统计 | 40 | 能量、过零率、平均幅值、峰值、能量变化 × 8种统计量 |
| MFCC | 26 | 13维 MFCC 的均值与标准差 |
| MFCC Delta | 26 | 13维一阶差分的均值与标准差 |
| 合计 | 92 | Python 与 C 端保持相同布局 |

- 采样率：16 kHz
- PCM：16-bit 单声道
- 模型窗口：1 秒
- 帧长：25 ms
- 帧移：10 ms

### 8.2 TinyMLP

```text
92维输入 → z-score → 64维全连接 + ReLU → 5维输出 → Softmax
```

| 指标 | 数值 |
|---|---:|
| 参数量 | 6,277 |
| Float32 参数及归一化数据 | 25,844 B |
| INT8 权重存储估算 | 7,029 B |
| 存储缩减 | 72.80% |

INT8 路径采用权重对称量化与运行时反量化，收益集中在存储空间，不涉及纯整数 MAC 加速。

## 九、可复现指标

评测以源录音为统计单位：390 条源录音切分为 1,950 个一秒窗口，同一源录音的窗口保持在同一折。background 有 230 条源录音，其余四类各 40 条，因此同时给出总体准确率与宏平均召回率。

训练使用混合增强：时间平移、音量缩放、时间拉伸和高斯噪声，并混合同训练折真实背景；背景只从训练折内取，测试折不参与。事件类使用 75% 增强样本，background 使用 25%，增强副本数为 4。

### 9.1 五折总体结果

| 条件 | Float 总体准确率 | Float 宏平均召回率 | INT8 总体准确率 | INT8 宏平均召回率 |
|---|---:|---:|---:|---:|
| Clean | 82.56% | 78.39% | 82.56% | 78.39% |
| 20 dB | 80.00% | 78.76% | 80.00% | 79.17% |
| 10 dB | 79.23% | 80.57% | 79.23% | 80.15% |
| 5 dB | 74.36% | 74.37% | 74.87% | 74.54% |

INT8 与 Float 结果基本一致，未观察到明显精度损失。

### 9.2 Clean 条件各类别召回率

| 类别 | 样本数 | Float Recall | INT8 Recall |
|---|---:|---:|---:|
| background | 230 | 86.96% | 86.96% |
| cough | 40 | 55.00% | 55.00% |
| glass_break | 40 | 85.00% | 85.00% |
| baby_cry | 40 | 85.00% | 85.00% |
| dog_bark | 40 | 80.00% | 80.00% |

### 9.3 5 dB 条件各类别召回率

| 类别 | Float Recall | INT8 Recall |
|---|---:|---:|
| background | 74.35% | 75.22% |
| cough | 47.50% | 47.50% |
| glass_break | 82.50% | 82.50% |
| baby_cry | 90.00% | 87.50% |
| dog_bark | 77.50% | 80.00% |

5 dB 下五类事件均可检出，其中 cough 召回率最低（47.50%）。真机端准确率、延迟、内存和功耗未测量。

## 十、OpenVela 端当前能力

### 已实现

- 16 kHz/16-bit/单声道麦克风三秒分段采集。
- 三个一秒窗口的 C 特征与 INT8 推理。
- 静音能量门限 `2.0e-5` 和事件阈值 `0.80`。
- 五类事件输出。
- LVGL 的 START、STOP、AUTO、状态、类别、置信度、能量和计数显示。
- 内存中保留最近 3 条事件摘要（当前 UI 尚未创建历史列表控件）。
- 自动启动偏好持久化和开机启动。
- 命令行 WAV 推理、噪声演示、合成模板关键词识别和可选 HTTP/MQTT 代码路径。

### 未实现或未验证

- 当前通过 `arecord` 分段写 WAV，不是无间隙 PCM 环形缓冲。
- 不支持重叠滑窗、多标签和事件起止时间定位。
- HTTP/MQTT 尚未接入实时 UI 工作线程。
- “救命、帮助、报警”关键词路径使用合成语音MFCC模板和DTW匹配，未接入实时UI，也没有真实录音独立评测；不属于五类模型的准确率指标。
- 没有持久化事件数据库或音频证据归档。
- 没有真机准确率、端到端延迟、峰值内存、功耗和长期稳定性数据。

更完整的边界见 [`docs/OpenVela端功能边界.md`](docs/OpenVela端功能边界.md)。

## 十一、AI Coding 使用说明

AI 工具参与了需求拆解、训练与交叉验证脚本完善、C/Python 一致性排查、板端问题定位、结果复核和提交材料整理；相关改动经过源码检查与运行验证。

本公开仓库不提供私人会话、账号信息或AI日志归集资料。AI工具并非应用运行时依赖，不参与设备端声音分类。

## 十二、许可证和数据来源

- 团队有权许可的项目源码采用 Apache License 2.0，见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE)；第三方内容仍适用其各自条款。
- ESC-50数据集整体许可为CC BY-NC 3.0，单条音频沿用上游列明的原始许可和署名；数据不随本仓库分发。
- 训练权重、生成模板、厂商组件和素材的边界见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`model/README.md`](model/README.md)。本仓库不作这些内容可无条件商用的保证。
- 实际历史依赖版本无法恢复的项如实标为未锁定，不以当前上游版本替代。资源披露已补充，不代表所有第三方授权已完成。

## 十三、证据索引

- [`results/python_cv/metrics.json`](results/python_cv/metrics.json)：五折指标、混淆矩阵与噪声鲁棒性原始结果。
- [`model/tiny_mlp_mfcc92_robust_simulator.npz`](model/tiny_mlp_mfcc92_robust_simulator.npz)：项目模型 NPZ。
- [`app/audiodetect/model_weights.h`](app/audiodetect/model_weights.h)、[`model_weights_int8.h`](app/audiodetect/model_weights_int8.h)：开发板权重头文件，由 `training/export_to_c.py` 从项目模型生成。
- [`simulator/`](simulator/)：主机 C 模拟器 INT8 推理实现与生成权重。
- [`docs/EVALUATION.md`](docs/EVALUATION.md)：当前v11实验口径和解释。
- [`docs/OpenVela端功能边界.md`](docs/OpenVela端功能边界.md)：OpenVela 端能力边界。
- [`docs/PUBLIC_RELEASE.md`](docs/PUBLIC_RELEASE.md)：本次公开整理的修改范围和排除内容。
