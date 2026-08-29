@echo off
rem ECAPAFlow: Port 7873 in der Windows-Firewall freigeben (einmalig).
rem Rechtsklick -> "Als Administrator ausfuehren"
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Bitte per Rechtsklick "Als Administrator ausfuehren" starten!
  pause
  exit /b 1
)
netsh advfirewall firewall add rule name="ECAPAFlow 7873" dir=in action=allow protocol=TCP localport=7873
echo.
echo Fertig! Jetzt am Handy neu laden: http://192.168.10.8:7873
pause
