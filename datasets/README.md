# 可追溯的CARLA数据

`carla_replay/case11`与`case07`各保存80张自采CARLA原RGB（双车共享同帧图像）、960张部件mask/边界，以及80帧双车预测/参考、相机和最新013/014结果。RGB共160张、407,992,772字节。SAM mask是现有提示和选择流程的派生产物，不是人工语义部件真值；SOURCE表面凸体也不是完整真实车壳或原可编辑资产。未携带CARLA地图或原GLTF/FBX材质纹理资产包、公司视频或真实道路私人录像。

`carla_replay/input_json/case11.json`与`case07.json`是轻量数值导出，合计664,647字节，保留全80帧姿态与原距离/接触结果，不含图像。其位置/速度指标可独立重算。

`carla_surface_sample/prepared_sample`来自个人历史SurfaceNet训练数据的episode0最早3步：6状态、35有效视图，含64×64真实RGB裁剪/native实例mask/像素坐标、标定、训练目标、独立参考几何与上游已有learned proposal。源状态/记录索引与源文件SHA在其 `input_json/SOURCE_SELECTION.json`，没有把GT冒充proposal，也没有重算该上游模型。样本是已见训练片段，仅用于协议及真实输入推理，不是独立测试集。

`carla_surface_sample/code/export_subset.py --source <原准备数据目录> --output <当前目的内新目录>`可从同协议数据重新导出，拒绝覆盖。原全量数据不在公共服务上，没有虚构下载链接。

`record_carla/code/record_existing_actors.py`连接已有CARLA世界，只记录指定现有actor的数值状态，不改变地图、物理设置或车辆控制，也不生成事故标签、RGB编码特征或SurfaceNet完整教师：

```text
python -B datasets/record_carla/code/record_existing_actors.py --actor-ids 46 47 --frames 80 --output datasets/record_carla/experiments/my_capture/output_json
```

需要自行运行匹配版本的CARLA服务与PythonAPI，将示例ID替换为当前世界中的actor ID；同步世界须由其他客户端推进。该连接脚本仅做CLI/语法检查，本轮未启动CARLA重新采集。关系学习缓存、SurfaceNet准备输入和这里的数值录制是不同协议，不能互换。

