@echo off
REM Wrapper for Windows Task Scheduler — runs 04_jdoc_sync.py during JDoc window.
REM Invoked daily at 02:00 local time (inside the 00:00-06:00 service window).

set "PROJECT=C:\Users\alex2\Desktop\vsCode\AiJudge"
set "PYTHON=C:\Users\alex2\AppData\Local\Programs\Python\Python39\python.exe"
set "LOGFILE=%PROJECT%\data\logs\jdoc_sync.log"

cd /d "%PROJECT%"

for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do set "STAMP=%DATE%_%%a%%b%%c"
echo [%STAMP%] === run start === >> "%LOGFILE%"

"%PYTHON%" scripts\04_jdoc_sync.py -v >> "%LOGFILE%" 2>&1

echo [%STAMP%] exit=%ERRORLEVEL% >> "%LOGFILE%"
exit /b %ERRORLEVEL%
