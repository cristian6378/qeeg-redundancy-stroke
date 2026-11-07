@echo off
REM Review-revision analyses of the MDPI redundancy article (Anaconda Prompt, stroke-eeg environment).
REM   run_review.bat check   quick check (minutes)      run_review.bat   full run
REM The motor-imagery onset (2.0 s) was confirmed with  python ..\pipeline\00_check_timing.py
cd /d "%~dp0"
if not exist ..\logs mkdir ..\logs
set MI_ONSET=2.0
if "%1"=="check" goto :check
echo [%time%] B3 task-matched windows
python B3_taskmatched.py --mi-onset %MI_ONSET% > ..\logs\rev_B3.txt 2>&1 || goto :fail
echo [%time%] B4 patient-level vs within-patient redundancy
python B4_hierarchy.py > ..\logs\rev_B4.txt 2>&1 || goto :fail
echo [%time%] B5 computation scaling
python B5_scaling.py > ..\logs\rev_B5.txt 2>&1 || goto :fail
echo [%time%] B6 bootstrap intervals
python B6_uncertainty.py > ..\logs\rev_B6.txt 2>&1 || goto :fail
echo [%time%] A3 hierarchical benchmark (about 1-3 h)
python A3_hierarchical_synthetic.py > ..\logs\rev_A3.txt 2>&1 || goto :fail
echo [%time%] done: results in ..\output (studyA3, studyB)
goto :eof
:check
python B3_taskmatched.py --mi-onset %MI_ONSET% --boot 5 > ..\logs\rev_check.txt 2>&1 || goto :fail
python B4_hierarchy.py --boot 5 >> ..\logs\rev_check.txt 2>&1 || goto :fail
python B5_scaling.py --reps 1 >> ..\logs\rev_check.txt 2>&1 || goto :fail
python B6_uncertainty.py --boot 20 --boot-redundancy 5 >> ..\logs\rev_check.txt 2>&1 || goto :fail
python A3_hierarchical_synthetic.py --fast >> ..\logs\rev_check.txt 2>&1 || goto :fail
echo check passed. Now run:  run_review.bat
goto :eof
:fail
echo A step FAILED. Send me the last lines of the newest file in the logs folder.
exit /b 1
