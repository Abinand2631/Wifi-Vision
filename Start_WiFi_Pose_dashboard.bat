@echo off
echo Waking up the WiFi-Pose AI...

:: 1. Navigate to your project folder first
cd /d "%~dp0"

:: 2. Launch the true Anaconda Prompt, activate the environment, and start the GUI
%WINDIR%\System32\cmd.exe /K "C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3 & conda activate densepose_env & streamlit run dashboard.py"