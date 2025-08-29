@echo off
echo ============================================
echo   Connecting to GitHub Repository
echo ============================================
echo.

echo STEP 1: Copy your GitHub repository URL from the browser
echo Example: https://github.com/YOUR_USERNAME/deepika-portfolio.git
echo.
set /p repo_url="Paste your GitHub repository URL here: "

echo.
echo STEP 2: Connecting your local code to GitHub...
git remote add origin %repo_url%
git branch -M main

echo.
echo STEP 3: Pushing your code to GitHub...
git push -u origin main

echo.
echo ============================================
echo   ✅ SUCCESS! Code pushed to GitHub!
echo ============================================
echo.
echo Next steps:
echo 1. Go to netlify.com
echo 2. Click "New site from Git"
echo 3. Connect to your GitHub repository
echo 4. Deploy automatically!
echo.
pause
