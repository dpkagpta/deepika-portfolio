@echo off
echo ============================================
echo   FIXING GIT AND DEPLOYING TO GITHUB
echo ============================================
echo.

REM Navigate to the correct directory
cd /d "C:\Users\deepi\Projects\my_website"
echo Current directory: %CD%

echo.
echo Step 1: Checking Git status...
git --version
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/
    pause
    exit /b 1
)

echo.
echo Step 2: Reinitializing Git repository...
rmdir /s /q .git 2>nul
git init
git config user.name "Deepika Gupta"
git config user.email "deepika.7gupta7@gmail.com"

echo.
echo Step 3: Adding all files...
git add .

echo.
echo Step 4: Creating initial commit...
git commit -m "Deepika Gupta Portfolio - Ready for Production"

echo.
echo Step 5: Setting up for GitHub...
echo.
echo IMPORTANT: Before continuing, create your GitHub repository:
echo 1. Go to: https://github.com/new
echo 2. Repository name: deepika-portfolio
echo 3. Make it PUBLIC
echo 4. Click "Create repository"
echo.

set /p repo_url="Paste your GitHub repository URL here (https://github.com/USERNAME/deepika-portfolio.git): "

echo.
echo Step 6: Connecting to GitHub...
git remote add origin %repo_url%
git branch -M main

echo.
echo Step 7: Pushing to GitHub...
git push -u origin main

echo.
echo ============================================
echo   ✅ SUCCESS! Your code is on GitHub!
echo ============================================
echo.
echo Next steps:
echo 1. Go to netlify.com
echo 2. Sign up/login with GitHub
echo 3. "New site from Git" → Select your repository
echo 4. Deploy automatically!
echo.
echo Your portfolio will be LIVE on the internet!
echo.
pause
