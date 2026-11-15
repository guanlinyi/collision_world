# 从目的到代码：两车接触故事

目的：沿着案例07、车0、第28帧，把“想判断什么 → 输入怎么走 → 哪段代码负责”连起来。完成标准是单HTML离线可读、可以按层深入和返回，真实代码与数据有来源，训练、求解、回放及其验证边界不混写。

当前入口：[从目的到代码_故事版.html](output_html/从目的到代码_故事版.html)。直接用浏览器打开即可，无需服务、模型或网络。默认每次显示一个层级；左侧目录可筛选，下方流程框继续展开，顶部和章节末尾可以回上一级。禁用JavaScript时也能顺序阅读全部内容。

维护文件：

- `input_json/story.json`：22节中文叙事、父子关系、流程和源码片段配置。
- `input_json/f19_counterexample.json`：历史第19帧接受路径的最小公开评价摘录，保留两份来源SHA；不依赖原工作目录运行，不含完整历史交接。
- `code/build_story.py`：仅使用Python标准库，按AST定位当前源码并提取原样片段，记录实际行数与文件／片段SHA。数据卡从仓库现有JSON择字段读取，生成后嵌在HTML中。
- `output_html/`：离线成品。参考页面保存在 `experiments/20261115_story/input_html/`，仅供风格阅读，不是运行依赖。

从仓库根目录生成一个**尚不存在**的输出文件：

```text
python -S -B docs/program_flow/code/build_story.py --output docs/program_flow/output_html/从目的到代码_故事版_新版本.html
```

不传 `--output` 时使用当前默认成品名。已有文件一律拒绝覆盖；更新请换新文件名。只检查叙事、数据、AST片段、内部链接与大小而不写文件：

```text
python -S -B docs/program_flow/code/build_story.py --check-only
```

输出路径限制在本目的目录内，临时验收版本可写在 experiments 中；不会启动模型、删除文件、连接外网或改算法。HTML不使用fetch、CDN、外部图片、字体或脚本，代码经过HTML转义，JSON脚本块中的小于号也会转义。成品代码与数据对应**生成时**的源码；修改算法后应生成新版本，不把旧片段当成当前源码。

本页整理已有证据：第28帧核心重新求解、缓存SAM候选重选及RGB结构重算属于已有运行证据；全片段是已有冻结回放，第19帧目标下降而三维误差变坏的根因仍未解决。本次编写解读没有重新训练模型或重新求解三维。

2026-11-15 网页验收：Chrome 实际浏览器在1440×1150和430×1000两种尺寸下，各通过59项交互与布局检查，覆盖22节、逐层跳转、返回、目录筛选及代码展开／收起。另已检查源码片段、内部链接、离线资源与JavaScript语法。结果见[网页验收](output_json/validation.json)，这些检查不代表新增算法精度验证。

静态检查可独立运行，输出文件必须尚不存在：

```text
python -S -B docs/program_flow/test/verify_story.py --html docs/program_flow/output_html/从目的到代码_故事版.html --output docs/program_flow/experiments/my_check/output_json/static_check.json
```
