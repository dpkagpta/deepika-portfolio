@echo off
echo ============================================
echo   Deepika Gupta Portfolio - Quick Deploy
echo ============================================
echo.

REM Check if Git is configured
git config --get user.name >nul 2>&1
if %errorlevel% neq 0 (
    echo Setting up Git configuration...
    set /p username="Enter your Git username: "
    set /p email="Enter your Git email: "
    git config --global user.name "%username%"
    git config --global user.email "%email%"
)

echo 1. Adding all changes to Git...
git add .

echo 2. Committing changes...
set /p message="Enter commit message (or press Enter for default): "
if "%message%"=="" set message="Content update - %date% %time%"
git commit -m "%message%"

echo 3. Checking if remote repository exists...
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo *** SETUP REQUIRED ***
    echo No remote repository configured.
    echo Please create a GitHub repository and run:
    echo git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
    echo git branch -M main
    echo git push -u origin main
    echo.
    pause
    exit /b 1
)

echo 4. Pushing to production...
git push origin main

echo.
echo ============================================
echo   ✅ DEPLOYMENT COMPLETE!
echo ============================================
echo Your website has been updated in production.
echo Visit your live site to see the changes.
echo.
pause
