@echo off
for /f "delims=" %%a in ('python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"') do @set CUDNN_FILE=%%a
for %%F in ("%CUDNN_FILE%") do set CUDNN_PATH=%%~dpF
set PATH=%CUDNN_PATH%\bin;%PATH%
