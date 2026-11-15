# 从准备好的二维观测估计三维状态

个人研究历史路线：64×64 RGB裁剪＋原生实例mask＋相机标定＋身份＋已保存上游proposal → SurfaceNet可见表面坐标/面分类/不确定性 → 多视图刚体拟合 → 因果状态更新 → 三维盒与覆盖/接触评价。它不是任意裸视频的端到端重建器；参考真值只由训练与显式评价读取。

`personal_checkpoint`携带个人历史SurfaceNet、uncertainty权重与资产尺寸/分类器和噪声参数。它们是原个人研究产物的只读副本，来源哈希见总清单，没有假装重新训练得到它们。`datasets/carla_surface_sample/prepared_sample`是原CARLA训练episode0最早3步、双车6状态/35有效视图的真实裁剪子集；已有proposal与native mask是强输入条件，且这是已见训练片段，不能用它证明泛化。

在根目录执行真实CARLA小样本推理：

```text
python -B estimate_geometry/code/run.py infer --input datasets/carla_surface_sample/prepared_sample --model estimate_geometry/personal_checkpoint --output estimate_geometry/experiments/my_carla_run/output_npz --device cpu --include-reference
```

生成表面坐标、原始/时序几何、contacts、评价及 `world.json`。开启根目录轻量网页后，用其文件选择器打开 `world.json`。加 `--include-reference`才会读取评价真值；保存的 `inference.json`明确预测输入和验证边界。提供身份中的缺失几何不当成“未接触”，几何盒重叠不当成事故标签。

训练程序恢复了原优化、uncertainty训练及train-only噪声校准。下面是**合成数学体的软件入口验收**，不是真实CARLA效果实验，也不是013/014重算。fixture故意给出正确proposal，不能用于性能报告：

```text
python -B estimate_geometry/code/make_smoke_fixture.py --output estimate_geometry/experiments/my_train/input_npz
python -B estimate_geometry/code/run.py train --input estimate_geometry/experiments/my_train/input_npz --output estimate_geometry/experiments/my_train/model --init estimate_geometry/experiments/my_train/input_npz/initialization.pt --asset-prior estimate_geometry/experiments/my_train/input_npz/assets.json --epochs 1 --uncertainty-epochs 1 --device cpu
python -B estimate_geometry/code/run.py infer --input estimate_geometry/experiments/my_train/input_npz --model estimate_geometry/experiments/my_train/model --output estimate_geometry/experiments/my_train/output_npz --device cpu --include-reference
```

省略 `--init`会从随机初始化训练，不能视为历史训练复现。真实完整训练还需要有代表性的训练episodes、目标和asset prior，特别是完整/裁边观测两种噪声分层；35视图子集主要用于真实推理与协议核查，不能代替完整训练集。原完整训练数据和初始化权重未全部携带，没有外部下载地址，不能把本地不存在的公共下载链接写成获取步骤。自行新采集后仍需生成native-mask身份、表面坐标教师和learned proposal等明确输入；本包采集脚本不会自动补齐这些步骤。

源代码在 `code/src`，正式入口为 `code/run.py`；目录与JSON/NPZ分层兼容原扁平数据和按实际扩展名分开的数据。所有CLI写入新目录，JSON拒绝覆盖。源码原有可选状态更新/缺失恢复仍是实验选项，不宣称改进了整体准确率。

013/014是后来的SAM部件/整车外缘投影优化候选，不是这里SurfaceNet网络的直接输出。[独立移植的最新R/BT8核心](refine_contours/README.md)已携带014/f28缓存输入，并重新构建SOURCE、选择对应及执行迭代，与冻结候选逐位一致。[单帧前端](observe_structure/README.md)现包含显式SAM权重入口与原RGB梯度逻辑，已验证缓存SAM候选选择与真实RGB梯度重算；神经SAM前向和全片段未重跑。原网页与掩码见[回放目的](../inspect_predictions/README.md)。

