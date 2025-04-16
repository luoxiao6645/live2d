@echo off
echo 正在使用官方PyPI源安装Live2D项目所需的依赖项...
echo 这可能需要一些时间，请耐心等待...

REM 使用官方PyPI源安装依赖项
pip install edge-tts==6.1.9 -i https://pypi.org/simple/
pip install numpy==1.24.3 torch==2.1.0 transformers==4.35.0 librosa==0.10.1 soundfile==0.12.1 -i https://pypi.org/simple/
pip install flask==3.1.0 flask-cors==5.0.0 -i https://pypi.org/simple/

REM 检查安装结果
echo.
echo 检查依赖项安装状态...
python -c "import numpy; import torch; import transformers; import librosa; import edge_tts; print('所有依赖项已成功安装！')" || echo 部分依赖项安装失败，请查看上面的错误信息。

echo.
echo 安装完成！
echo 现在您可以运行 python server.py 启动服务器。
pause
