@echo off
echo 正在安装Live2D项目所需的依赖项...
echo 这可能需要一些时间，请耐心等待...

REM 安装基本依赖项
pip install -r requirements.txt

REM 检查安装结果
echo.
echo 检查依赖项安装状态...
python -c "import numpy; import torch; import transformers; import librosa; print('所有依赖项已成功安装！')" || echo 部分依赖项安装失败，请查看上面的错误信息。

echo.
echo 安装完成！
echo 现在您可以运行 python server.py 启动服务器。
pause
