# 从现有CARLA世界记录输入

目标：导出现有车辆数值状态及地图静态物体盒，给后续数据准备提供可追溯原始材料。这里不会生成完整训练教师、部件提示、RGB编码特征或事故标签。

`code/record_existing_actors.py`只记录指定现有actor，不修改世界设置、车辆控制或地图。操作说明见[数据总览](../README.md)。

`code/export_carla_static_geometry.py`从匹配的已运行CARLA世界读取环境物体盒，保留原`mask2world-carla-static-geometry-0.1`协议。默认类别为Walls、Fences、Poles、TrafficSigns，输出地图名、类别计数、物体ID、世界坐标下的location、rotation、extent及8角点。只读获取，不加载地图、不生成或销毁actor。

```text
python -B datasets/record_carla/code/export_carla_static_geometry.py --host 127.0.0.1 --port 2000 --output-dir datasets/record_carla/experiments/my_map/output_json
```

`--output-dir`必须是尚不存在的新目录；导出`static_geometry.json`，独占创建，拒绝覆盖。需要与服务端匹配的CARLA PythonAPI和运行中的世界。原静态导出逻辑来自早期公司阶段源码，仅作本地待权属与许可审核的整理，不因此获得重新发布许可；来源及改动哈希见[SOURCE_MANIFEST_STATIC_EXPORT.json](SOURCE_MANIFEST_STATIC_EXPORT.json)。

此文件可供[原接触教师生成器](../../learn_relations/README.md)读取，必须与捕获地图一致。单车静态教师原算法按最近中心选物体，物体盒不是实际车辆网格或碰撞器。仅动态双车数据可显式提供同schema且`objects: []`的文件，不能用空地图冒充静态接触数据。

本轮仅执行语法解析与`--help`，退出0；未连接CARLA服务或实采新地图。它也不会把数值录制自动转成SurfaceNet或旧关系学习的完整准备输入。
