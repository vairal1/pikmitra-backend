@echo off
echo 🚀 Starting Pikmitra Python Backend with PyTorch MobileNetV2...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

REM Check for model files
echo 🔍 Checking for PyTorch MobileNetV2 model...
if exist "..\model\mobilenet_v2_1.0_224-plant-disease-identification\pytorch_model.bin" (
    echo ✅ Found PyTorch MobileNetV2 model
) else (
    echo ⚠️ PyTorch model not found at expected location:
    echo    ..\model\mobilenet_v2_1.0_224-plant-disease-identification\pytorch_model.bin
    echo    The app will use mock predictions without a model.
)

REM Check for .env file
echo 🔍 Checking for environment file...
if not exist ".env" (
    echo ⚠️ .env file not found. Please create a .env file in the 'backend' directory with the following content:
    echo.
    echo JWT_SECRET_KEY="your-super-secret-and-long-random-string-for-jwt"
    echo NEWS_API_KEY="your_news_api_key_here"
    echo.
    echo ❌ Halting startup. Please create the .env file and run again.
    pause
    exit /b 1
)
echo ✅ Found .env file.

REM Set environment variables
set FLASK_ENV=development

echo.
echo ✅ Setup complete! Starting Flask server...
echo 🌐 Backend will be available at: http://localhost:5000
echo 📊 Health check: http://localhost:5000/health
echo.

REM Start Flask app
python app.py

pause
