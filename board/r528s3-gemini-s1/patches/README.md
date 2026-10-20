# LVGL 小屏缓冲适配

`lvgl-aligned-buffers.patch` 源于实际验证的 SDK LVGL 提交
`9465c48a115dcd6f00f643b7fadae81e1da2f9d1`，其父提交是
`0f2a49f588505a00e8b46e25a34581c87291a62a`。补丁只修改
`src/drivers/nuttx/lv_nuttx_lcd.c`：分配遵守 `LV_DRAW_BUF_ALIGN=64`，
保存并释放原始分配指针，避免把 LVGL 的 draw-buffer 描述符当分配地址释放。

`scripts/apply_board_overlay.sh` 在该 R528 SDK 的 LVGL checkout 内进行
apply/reverse-check，实现重复执行检查；未知源码冲突时退出，不猜测解决。
它不适用于 goldfish framebuffer 配置。构建入口保存 LVGL 实际 HEAD 和
未提交 patch，使 manifest -r 与工作区差异共同描述依赖。原始构建快照和
补丁见 `results/board-20261020/` 与本地固件交付目录。
