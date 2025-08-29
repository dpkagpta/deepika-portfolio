@echo off
echo ============================================
echo   FORCE UPDATING LIVE WEBSITE
echo ============================================
echo.

cd /d "C:\Users\deepi\Projects\my_website"

echo Current files in repository:
dir /b

echo.
echo Force adding ALL current files...
git add .
git add --all

echo.
echo Creating new commit with current sophisticated website...
git commit -m "FORCE UPDATE: Complete sophisticated portfolio with all features" --allow-empty

echo.
echo Force pushing to GitHub (this will update your live Netlify site)...
git push origin main --force

echo.
echo ============================================
echo   ✅ DONE! Your live website is updating!
echo ============================================
echo.
echo 🚀 Netlify will automatically rebuild in 1-2 minutes
echo Check your live site in a few minutes!
echo.
pause
