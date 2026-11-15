# 核对观测、姿态与接触证据

完整模式是原网页的便携整理：根目录 `python -S -B start_full_replay.py`，打开8883端口的 `/cases/11/` 或 `/cases/07/`。原图、部件mask/边界、原SOURCE表面凸体、相机、预测/参考、历史方法和最新013/014均在项目内；没有CDN、SAM服务、外部磁盘回退或后台推理。

轻量模式为 `python -S -B start_replay.py`，端口8882。只读数值小样本，支持拖动姿态示意、播放、跳红格和打开SurfaceNet导出的 `world.json`。轻量数学框不参与原凸体距离计算，也不代替原图核查。

`code/verify_full_replay.py`启动临时本地HTTP服务，检查两个案例各80帧160姿态、1120个PNG URL，并逐个核对RGB传输SHA。`code/verify_replay.py`独立重算位置/相邻速度误差，保留07的19/33速度红格；本轮最大差在5e-16量级。两段冻结接触FP/FN为0不是新的事故识别成绩。

完整模式的Three.js MIT notice保留在 `web_rich/input_txt`；无许可副本的字体未搬运，改用系统字体。数据内工作站绝对路径已改为不参与运行的来源说明，源码与数值变化记录见总来源清单。浏览器实际画面检查范围及阻塞见[验收记录](../docs/VALIDATION.md)，不以HTTP成功冒充视觉验收。

