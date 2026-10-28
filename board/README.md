# R528S3 Gemini S1 板级集成

当前板级配置面向R528S3 Gemini S1、128 MiB DDR和板载2.8寸ILI9341小屏。配置包含音频采集、触摸、LVGL、Tiny MLP应用和开机自启。`configs/nsh/defconfig`与`src/etc/init.d/rcS.nsh`为应用增量，LVGL显示缓冲补丁位于`patches/`。

## 构建

在独立OpenVela SDK副本中准备应用Git检出及`apps/audiodetect`链接，应用版本与主仓库一致。在SDK根目录创建隔离标记后运行：

```bash
touch SDK/.audio-sentinel-isolated
bash scripts/build_r528s3_image.sh SDK OUTPUT
```

SDK、OUTPUT为本地路径。构建器检查应用版本、模型及配置，执行板级overlay、交叉编译和官方打包，输出镜像、AP固件哈希、工具链版本与实际`repo manifest -r`。现成的32位dragon运行库是打包前提。

脚本只构建与打包，烧录由使用者单独执行。R528的显示和内存配置用于真板；完整goldfish模拟器选择其自身配置。当前已测镜像的版本和NAND运行问题见[真板结果](../results/board-mlp-20261027/README.md)。
