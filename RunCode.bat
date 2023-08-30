@echo off
SET DRIVE_LETTER=%cd:~0,2%
SET BASE_DIR=%DRIVE_LETTER%\Software

SET PATH=%BASE_DIR%\PortableGit\bin;%PATH%
SET PATH=%BASE_DIR%\miniconda3;%PATH%
SET PATH=%BASE_DIR%\miniconda3\Scripts;%PATH%

SET ENV_PATH=%BASE_DIR%\miniconda3\envs\CV 

SET PATH=%ENV_PATH%\lib\site-packages\nvidia\cudnn\bin;%PATH%
SET PATH=%ENV_PATH%\lib\site-packages\nvidia\cublas\lib\x64;%PATH%

START "" %BASE_DIR%\VSCode-win32-x64-1.81.1\Code.exe 
