# RK3562 端侧 AI 视觉检测终端

基于迅为/topeet RK3562 开发板的实时端侧视觉终端：USB 摄像头 → V4L2 → CPU 预处理 → RKNN NPU（YOLOv5s INT8）→ 后处理/NMS → IoU 跟踪与人数统计 → 800×1280 屏显 + MJPEG 推流，Buildroot 上电自启。

## 1. 硬件与环境

| 项 | 规格 |
|---|---|
| 开发板 | 迅为 RK3562，Buildroot，glibc 2.36，RKNN runtime 1.5.0 |
| NPU | RK3562 NPU，YOLOv5s INT8（640×640）单帧推理 ~50 ms |
| 摄像头 | USB UVC `0c45:64ab`，YUYV 640×480@30，节点 `/dev/video18` |
| 屏幕 | 800×1280 MIPI DSI，framebuffer `/dev/fb0`（XRGB8888，stride 3200） |
| 网络 | RTL8723du USB WiFi（2.4 GHz），wpa_supplicant + udhcpc |
| 交叉工具链 | gcc-arm-11.2-2022.02（aarch64-none-linux-gnu） |
| 模型转换 | rknn-toolkit2 1.5.0（Python 3.8，Ubuntu 虚拟机） |

## 2. 数据流与线程模型

三线程 + 三槽位的流水线，任一槽位满时丢最旧帧，保证端到端低延迟：

1. **采集/预处理线程**：V4L2 MMAP 4 缓冲 → YUYV→RGB（手写查表）→ letterbox 到 640×640（灰度填充）→ 写槽位。
2. **推理线程**：`rknn_inputs_set` → `rknn_run` → `rknn_outputs_get`（int8 + zp/scale 反量化）→ 解码 + NMS（0.35/0.45）→ person IoU 跟踪 → 写槽位。
3. **输出线程**：按屏尺寸缩放粘贴 → 画检测框/类别/置信度/HUD → 写 `/dev/fb0`；同时 turbojpeg 编码 → MJPEG HTTP 推送。

架构图见 [`docs/architecture.svg`](docs/architecture.svg) / [`docs/architecture.png`](docs/architecture.png)。

## 3. 实测结果

单帧 NPU 推理（W1）：**49.07 ms，20.4 fps**。

600 帧相机链路（W3，`/dev/video18`，INT8）：

| 模型 | 端到端 fps | infer p50 / p95 | conv p50 | post p50 | total p50 / p95 | 模型大小 |
|---|---|---|---|---|---|---|
| yolov5s INT8（200 帧重标定） | **16.45** | 50.57 / 51.11 ms | 17.94 ms | 8.59 ms | **60.83 / 61.73 ms** | 7.64 MB |
| yolov5s INT8（原校准） | 16.44 | 50.48 / 50.92 ms | 18.01 ms | 8.61 ms | 60.85 / 62.8 ms | 7.64 MB |
| yolov5s FP16 | 0.37 | 118.36 / 137.07 ms | 3.98 ms | 2544 ms | 2714 ms | 14.89 MB |

- FP16 输出为 float16，现有 INT8 解码路径无法使用；即使补 float 后处理，推理也是 INT8 的 2.3 倍。**RK3562 上选择 INT8**。
- 重标定与原模型的 200 帧逐帧对比：person 人数一致率 **95.5%**（191/200），最高置信度平均差 **0.0166**，验证重标定没有破坏模型。
- 上电自启后连续运行 23,300+ 帧，稳定 **16.6 fps**（相机本身上限 16–22 fps，受室内光照限制）。
- 仓库自带 `model/yolov5s_int8_calib.rknn`（7.64 MB）与 COCO 标签；ONNX/权重来自 Ultralytics YOLOv5（GPL-3.0），转换脚本见 `convert_calib.py`。

## 4. 构建

```bash
cmake -S . -B build-arm -DCMAKE_TOOLCHAIN_FILE=$PWD/toolchain-aarch64.cmake
cmake --build build-arm -j4
```

`rknn_stub.c` / `tj_stub.c` 编译出仅供链接的桩库 `librknnrt.so`、`libturbojpeg.so.0`，运行期由板端真实库提供，交叉编译时无需拷贝厂商 `.so`。

## 5. 部署

```text
板端 /root/demo/
├── rknn_cam                       # 主程序
├── model/yolov5s_int8_calib.rknn  # INT8 模型
├── model/coco_80_labels_list.txt
├── calib/                         # 200 帧校准集（可选）
└── /etc/init.d/S99rknn_cam        # 上电自启脚本
```

`S99rknn_cam` 依次执行：静默内核日志（`dmesg -n 1`）→ 停 QLauncher/weston → 等待摄像头 → 启动 `rknn_cam` 上屏 → 拉起 wpa_supplicant（`rtw_power_mgnt=0 rtw_ips_mode=0` 关省电）→ udhcpc 获取 IP。WiFi 配置在 `/etc/wpa_supplicant.conf`（不入库）。

## 6. 运行模式

```bash
# 实时：上屏
./rknn_cam /dev/video18 model/yolov5s_int8_calib.rknn 0 0 /dev/fb0

# 实时：上屏 + MJPEG（浏览器打开 http://<板 IP>:8081/stream）
./rknn_cam /dev/video18 model/yolov5s_int8_calib.rknn 0 8081 /dev/fb0

# 采集校准集：前 200 帧 JPEG 落到 calib/
./rknn_cam /dev/video18 model/yolov5s_int8_calib.rknn 200 0 none dump

# 离线评估：逐帧 CSV + 汇总
./rknn_cam eval model/yolov5s_int8_calib.rknn calib 200

# 文件服务器（取日志/CSV）
./rknn_cam serve 8090 .
```

参数位：`<video> <model> <frames(0=无限)> <mjpeg port(0=关)> <fb 或 none> [swap] [dump]`。

## 7. 关键实现点

- **INT8 反量化**：从 `RKNN_QUERY_OUTPUT_ATTR` 读 zp/scale，`(int8 - zp) * scale` 后解码三个尺度（80/40/20），不依赖固定阈值。
- **预处理**：YUV→RGB 用整数查表，letterbox 保持宽高比并记录缩放/偏移，检测框映射回屏幕坐标。
- **帧率瓶颈**：NPU 50 ms 占端到端 61 ms 的 83%；下一步 416×416 输入可近似减半，RGA 硬件前处理可再省 ~18 ms（板端已有 `/dev/rga`，未实现）。
- **检测抖动**：单帧 YOLOv5s 对行人稀疏，直接逐帧计数会跳变；用 IoU 跟踪 + 命中确认 + 3 s/200 px 防抖后人数稳定。

## 8. 已知限制

- 校准集为单一室内场景的 200 帧自拍，未做带标注的 mAP 评估。
- 前处理仍为 CPU（~18 ms/帧），RGA/NV12 零拷贝未做。
- 当前模型输入 640×640；416 版本需重新导出权重（仓库未含 `.pt`）。
- WiFi 为 USB 2.4G 单天线，弱信号下关联较慢；已通过关闭驱动省电缓解。