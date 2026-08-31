@echo off
call D:\anaconda\Scripts\activate.bat wham_gmr
set OUTPUT_ROOT=output/test_run
set ROBOT=unitree_g1
set RECORD_GMRVIDEO=1
set RECORD_WHAMVIDEO=0
set VIDEO=video_input/dateset_vedio.mp4
powershell -ExecutionPolicy Bypass -File run.ps1
