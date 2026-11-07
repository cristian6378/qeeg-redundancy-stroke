@echo off
REM Runs Studies A-C of the MDPI redundancy article in order (Anaconda Prompt, stroke-eeg environment).
cd /d "%~dp0"
if not exist ..\logs mkdir ..\logs
for %%s in (A1_selection_synthetic A2_analyze_synthetic B1_extract_variants B2_measurement_properties C1_clinical_severity) do (
  echo [%time%] %%s
  python %%s.py > ..\logs\mdpi_%%s.txt 2>&1 || goto :fail
)
echo [%time%] done: results in ..\output
goto :eof
:fail
echo A step FAILED. Open the newest file in the logs folder and send me its last lines.
exit /b 1
