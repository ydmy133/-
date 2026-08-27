@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cl /nologo /std:c11 /W4 /I..\Core\Inc test_com_proto.c ..\Core\Src\com_proto.c /Fe:test_com_proto.exe > cl.log 2>&1
if errorlevel 1 (
  echo COMPILE FAILED
  type cl.log
  exit /b 1
)
test_com_proto.exe
exit /b %errorlevel%
