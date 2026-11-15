# 本轮验收范围

整理日期：2026-11-15。验证以Windows上的项目授权Python环境为准，已测Python3.12.0、NumPy2.4.6、SciPy1.17.1、Torch2.7.1+cu126。以下小型学习计算使用CPU，没有新训练GPU大模型；全新安装、Linux与Orin尚未验收。

## 已执行并通过

| 入口或材料 | 实际验证与结果 | 证据及边界 |
| --- | --- | --- |
| 关系/事件学生训练与推理 | 合成输入的微型软件验收：3条记录各8窗口，真实优化2轮，保存/加载checkpoint并推理，子进程均退出0；去除所有教师字段后预测最大差0 | [acceptance.json](validation/output_json/relation_smoke.json)。不含V-JEPA编码器或真实CARLA训练效果 |
| 个人SurfaceNet训练与推理 | 合成数学体的小型软件验收：表面网络1轮、uncertainty1轮，保存后CPU推理及导出，退出0 | [inference.json](validation/output_json/surface_smoke_inference.json)及相邻训练产物。fixture给出正确proposal，不代表现实输入或性能 |
| 个人历史权重＋真实CARLA子集 | 35个有效视图、双车6状态的真实裁剪/native mask/标定/已有proposal，CPU推理退出0 | [inference.json](validation/output_json/surface_real_inference.json)。来源为已见训练episode最早3步，不能证明泛化；预测输入不读深度或参考真值，独立评价另行处理 |
| 轻量回放核验 | `python -S -B inspect_predictions/code/verify_replay.py`退出0 | 重算已有冻结数值和协议，不等于重新做RGB感知 |
| 完整网页资源核验 | `python -S -B inspect_predictions/code/verify_full_replay.py`退出0；每个案例1120个PNG URL、两例共2240个，通过真实HTTP请求；RGB传输SHA与项目内原图一致 | 每例80帧、双车160姿态；共享同帧RGB只保存一份。资源可访问与图像hash一致不替代浏览器交互验收 |
| 原接触教师恢复 | 4组合成CARLA格式输入，原源码与迁移函数的特征及四列目标逐元素一致；绝对路径原缓存与相对路径新CLI一致，退出0 | [acceptance.json](validation/output_json/contact_teacher_consistency.json)。保持原XY凸包/z间距、最近中心静态选择及逐记录帧速度变化定义，不证明物理碰撞器真值 |
| CARLA采集/静态导出入口 | CLI/语法检查通过，静态导出`--help`退出0 | 本轮未连接CARLA服务，未运行新采集、地图导出或完整数据教师准备 |

从根目录重做软件验收必须使用新输出目录，具体命令、依赖与输入协议见[关系学习说明](../learn_relations/README.md)和[几何说明](../estimate_geometry/README.md)。有真实原输入的历史权重推理与合成输入的小训练是两种不同验证，不能合并称为历史训练已复现。

## 浏览器画面

本轮用Chrome headless可执行程序实际加载并截图案例11，以及案例07的`#frame=28&car=0&method=case07_far_exterior_bt8`，两个进程均退出0。整理代理已查看[案例11截图](validation/output_png/case11.png)及[案例07截图](validation/output_png/case07_latest014.png)，看见原RGB、mask、三维视图和相应最新方案。截图与进程日志保存在该实验目录；浏览器profile与缓存属于本地临时材料，不进入git。

新增前端验收：缓存模式重新选择candidate2/12999像素，三个候选摘要及mask逐像素一致；真实RGB和新mask重算463点/248候选，所有字段与数值一致，三条命令均退出0。证据见[候选选择](validation/output_json/sam_cache_selection.json)与[RGB结构重算](validation/output_json/rgb_edges_consistency.json)。显式SAM仓库/权重/device接口已提供，新SAM神经前向未运行。

这确认了两个实际页面状态的静态显示。尚未逐项操作自动播放、拖动旋转、全部按钮或全部帧，也没有宣称Orin浏览器通过。页面可以展示既有结果，不说明其所依赖的原训练、SAM提取或最新求解已完整重算。

## 当前未知与解释边界

013/014最新R/BT8求解核心已独立移植，014/c0/f28从包内原SOURCE三角形重新构建每次姿态几何、读取真实缓存RGB有限边并重新选初始anchors/刷新对应、执行原内层求解和回溯。CPU实际89.55秒、退出0，6轮接受；中心差0m、转角差0°、代价差0，状态/轮数一致。新预测先落盘后才读冻结候选，未读QUERY真值。证据：[原acceptance](validation/output_json/latest014_frame28.json)、[直立角只读复核](validation/output_json/latest014_upright.json)。这是缓存上游观测模式的一帧复现；SAM神经前向及整段160帧未重跑；单帧候选选择和photometric生成已重新执行，不能冒称任意视频端到端。

两个80帧片段的几何接触验收仅覆盖这两个已有片段；案例07仍有第19、33帧速度红格。准确资产、标定、地面、SOURCE、身份、已见输入以及部分方案的未来RGB模板均是必须保留的前提，不冒称在线裸视频感知。几何盒接触不等于事故识别；合成优化成功、软件退出0或视觉平滑都不证明真实道路准确性。

未做：完整CARLA训练成绩复现、V-JEPA官方权重RGB提取重跑、SAM权重推理重跑、真实道路事故验证、跨场景泛化、概率可信度评估或公开服务部署。数据与第三方条件见[分发说明](RELEASE.md)。

## GitHub展示截图

发布展示使用实际本地网页：案例07第41帧选择最新014，预测与参考均接触；案例11第22帧选择最新013，均未接触。[截图及采集记录](showcase/output_json/capture_provenance.json)仅显示这两个实际页面状态，不代表全按钮交互或新平台验收。
