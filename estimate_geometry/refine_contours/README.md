# 最新R/BT8轮廓求解核心

这是013/014所复用的真实显式求解核心的独立移植。案例为014的 `joint_fresh_07/c0/f28`：缓存的RGB有限边/法向＋给定旧prior → 原SOURCE三角形投影轮廓 → 重新选择初始anchors/刷新2D↔3D对应 → 原内层least_squares与最多8轮刷新后代价回溯。原冻结候选只在预测落盘后用于核对，不用于初始化或选轮次。

根目录执行（需要本目的requirements）：

```text
python -B estimate_geometry/refine_contours/code/infer_frame.py --output estimate_geometry/refine_contours/experiments/my_new_replay --verify-frozen
```

本机CPU实际重算89.55秒、退出0，接受6轮，最终中心、转角和目标与冻结014逐位一致，三项差均0。每次尝试的SOURCE几何、对应、准入、内层状态和回溯证据写入新目录。未复用旧拟合结果或旧SOURCE pool，没有读取QUERY真值。原receipt保留；后来只对已保存预测新增直立转角核对receipt，没有重复求解。

输入契约：`input_json/safe_inputs.json`提供本帧旧prior、旧过滤基线、相机内外参和图像索引；`support.json`提供给定地面、支撑高度和SOURCE原点；`observation.json`是上游SAM粗ROI＋实际RGB梯度得到的有限边、法向和准入状态；`geometry_source/input_npz`是原SOURCE三角形和精确顶点/边/面索引。原图复用 `datasets/carla_replay/case07/input_png/rgb/rgb_028.png`。这些强输入不冒称由当前RGB独立感知得到，缓存观测也不冒充人工语义真值。

坐标为LAB xyz米，后3量是**度制旋转向量**。当前`Geometry.full`限定 `[0,0,rz]`，只优化x/y/直立z轴朝向，高度来自给定地面。核对拒绝非零的前两个旋转分量；不得当任意Euler姿态使用。迁移核对1e-8m/1e-7°/1e-8目标值容差仅允许浮点重复噪声，远小于像素尺度，不是现实几何准确率门限。

科学实现由原`fit_frame`、SOURCE投影/切锥证书、有限边匹配和`refresh`物化；保留原预算3→8的AST改动以及R端点/精确union拓扑修复、原权重、缺失罚项、回溯和停止规则。`SOURCE_TRANSFORMATION.json`与`SOURCE_MANIFEST.json`记录来源。运行时不依赖旧E/G/F项目或跨实验动态import。

[单帧前端](../observe_structure/README.md)现已提供显式SAM权重入口、候选准入与原RGB梯度生成；缓存候选选择和真实RGB梯度重算全部字段与冻结输入一致。使用 `--observation <新frame_28.json>` 可接入本求解入口，SAM神经前向未实跑。原十帧NCC/attempt审计包装未搬入；其原始入口是原个人项目 `diagnose_geometry_evidence/experiments/20261115_case07_far_observation` 的 `attempt3`，完整原路径和哈希在来源清单。当前提供缓存模式；重做新观测需要官方SAM权重、原提示/ROI和相应RGB梯度选择/准入流程，不能将任意mask或另一帧观测代入冒称等价。013/014全片段及历史结果仍完整回放，本轮仅重新求解这一帧，不宣称重跑SAM前端、全160帧或真实事故评价。已知资产、身份、标定、地面与prior仍是限制。
