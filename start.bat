@echo off
REM Double-clickable launcher: runs start.ps1 (bypasses execution policy).
REM Does NOT touch LM Studio, MySQL, or ChromaDB.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" %*
