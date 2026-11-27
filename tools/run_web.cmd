@echo off
rem 一键起 mjlab 总览（自动带起训练工作台 :8765 和曲线终端 :8766）
rem 用法：双击这个文件；或者 run_web.cmd --no-autostart / --host 0.0.0.0
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "tools\lgdog.py" web %*
) else (
  python "tools\lgdog.py" web %*
)
if errorlevel 1 pause
