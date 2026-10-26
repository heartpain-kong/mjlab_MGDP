"""mjlab 总览：一个本地页面看本机状态，并把工作台 / 曲线终端 / TensorBoard 串起来。

    python lgdog.py web            # 端口 8770，自动带起工作台和曲线终端
    python -m src.portal --no-autostart

只读系统状态 + 管理自己启动的子服务；不联网、不写训练产物。
"""
