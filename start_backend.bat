@echo off
echo 🚀 Starting Pikmitra Flask Backend...
echo.

cd /d "%~dp0"

echo 📥 Installing basic requirements...
pip install flask flask-cors pillow numpy requests werkzeug

echo.
echo ✅ Starting Flask server...
echo 🌐 Backend will be available at: http://localhost:5000
echo 📊 Health check: http://localhost:5000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python -c "from app import app; app.run(host='0.0.0.0', port=5000, debug=True)"
