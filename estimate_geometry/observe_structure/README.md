# 单帧SAM候选与RGB结构前端

此入口为014的07/车0/第28帧保留原科学逻辑：给定frame27同场景seed及第28帧已平移的ROI/提示 → 当前RGB的SAM三候选 → 3/4正点、无负点、至少seed面积一半的准入 → 合格候选中最高SAM自评分 → 原PCA法向/真实RGB单峰梯度候选。没有使用姿态、SOURCE匹配代价或QUERY真值选择mask。

缓存模式可以独立重做候选选择、mask导出和真实RGB梯度，再将观测显式接入最新求解：

```text
python -B estimate_geometry/observe_structure/code/observe_frame.py --output estimate_geometry/observe_structure/experiments/my_sam_cache --cached-candidates estimate_geometry/observe_structure/input_npz/frame28_sam_candidates.npz --verify-frozen
python -B estimate_geometry/observe_structure/code/observe_rgb_edges.py --frames 28 --car 0 --run reliable_exterior --whole estimate_geometry/observe_structure/experiments/my_sam_cache/output_json/observations.json --output-root estimate_geometry/observe_structure/experiments/my_edges
python -S -B estimate_geometry/observe_structure/code/verify_edges.py --observed estimate_geometry/observe_structure/experiments/my_edges/output_json/reliable_exterior/frame_28.json --output estimate_geometry/observe_structure/experiments/my_edges/output_json/acceptance.json
python -B estimate_geometry/refine_contours/code/infer_frame.py --observation estimate_geometry/observe_structure/experiments/my_edges/output_json/reliable_exterior/frame_28.json --output estimate_geometry/refine_contours/experiments/my_solved_frame --verify-frozen
```

均须新输出目录，命令从项目根执行。前三条本轮退出0：重新选择candidate2、12999像素，三候选摘要与mask逐像素一致；重新读取RGB和新mask得到463个轮廓点、248个photometric candidates，最终JSON所有字段与全部数值和后端原冻结输入完全相同。后端此前已用该完全相同观测独立重算89.55秒通过；没有再对同一观测重复89秒计算。前端不依赖GPU或SAM安装，需NumPy/OpenCV/Pillow/SciPy。

真实SAM前向源码也已提供，用显式参数替换第一条缓存模式：

```text
python -B estimate_geometry/observe_structure/code/observe_frame.py --output estimate_geometry/observe_structure/experiments/my_new_sam --sam-root path/to/official_sam2_repo --checkpoint path/to/sam2.1_hiera_large.pt --device cuda
```

代码保留原Hydra `sam2.1/sam2.1_hiera_l.yaml` 构建、checkpoint加载和SAM2ImagePredictor参数。外部代码/权重从[官方SAM2仓库与checkpoint说明](https://github.com/facebookresearch/sam2#download-checkpoints)获取，按官方依赖准备环境并记录commit、包版本及checkpoint SHA。本轮未下载或分发SAM权重，也没有运行新的SAM GPU/CPU神经前向；`--device cpu`虽有接口，尚未验收性能或数值一致性。缓存验收不能证明新的官方代码版本产生同样mask。

输入是特定同场景的已保存seed、提示和NCC平移ROI；十帧NCC跟踪、attempt审计包装及自动新目标发现没有移植为通用入口。frame27模板对早于27的帧有未来信息，本例28不据此增加泛化声明。mask仍是提示条件下的ROI/候选，不是人工语义真值。`whole_observer.py`来自原整车观测源码，`observe_rgb_edges.py`保留原单峰/竞争峰数值逻辑；仅重定位资源、抽出显式backend并应用原case07的3/4准入，来源见 `SOURCE_MANIFEST.json`。
