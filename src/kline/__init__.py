"""TensorBoard 曲线终端：把标量曲线当行情看，按窗口合成蜡烛图。

这个包只做三件事：
  1. 找到 ``logs/rsl_rl/<实验>/<run>/events.out.tfevents*`` 并读成标量序列（``store``）；
  2. 把序列按步长切窗口、合成开高低收（``aggregate``），供 K 线图和导出使用；
  3. 起一个只监听 127.0.0.1 的网页服务（``server``）。

除 ``tensorboard`` 是可选的读取依赖外，全部使用 Python 标准库；没有
tensorboard 时服务照样起得来，只是会在页面上明确报「读不了 event 文件」。
"""
