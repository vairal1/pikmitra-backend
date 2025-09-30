# Render Deployment Configuration for PikMitra Backend

## Repository Information
- **GitHub Repository**: https://github.com/vairal1/pikmitra-backend
- **Branch**: main

## Web Service Configuration

### Basic Settings
- **Name**: pikmitra-backend
- **Root Directory**: `` (leave empty since this IS the backend directory)
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --workers=4 --bind=0.0.0.0:$PORT`

### Environment Variables
Add the following environment variables in Render dashboard:

```
DATABASE_URL=postgresql://postgres:Vairal@182005@db.ngyvswlpwyipdmhpjegk.supabase.co:5432/postgres
SUPABASE_URL=https://ngyvswlpwyipdmhpjegk.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5neXZzd2xwd3lpcGRtaHBqZWdrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgxMzc4MjUsImV4cCI6MjA3MzcxMzgyNX0.3rMhj6AO0vIKM--5J6_p3RgLtVb9PIlfHvzlaNFWoAk
JWT_SECRET_KEY=2LyxDGLw6Wp/cgItc0mcCtJqW0xx6NwAGhctPHdqSag4st2m4IYz4ThG9Wtv0r6NGnSUxSbTIPV4MHWjdwkv1w==
MODEL_PATH=model/mobilenet_v2_1.0_224-plant-disease-identification
FLASK_ENV=production
```

### Auto-Deploy
- Enable auto-deploy from the main branch

## Features Verified
✅ PyTorch CPU installation with extra index URL
✅ ML model files included (9.3MB pytorch_model.bin)
✅ All required dependencies in requirements.txt
✅ Gunicorn production server configuration
✅ Environment variables template
✅ Database and Supabase configuration

## Deployment URL
Once deployed, your backend will be available at:
`https://pikmitra-backend.onrender.com`

## API Endpoints
- POST /predict - Plant disease detection
- GET /market-prices - Market prices data  
- GET /weather - Weather information
- GET /schemes - Government schemes
- GET /news - Agricultural news

## Model Information
The ML model is a MobileNetV2-based plant disease identification model:
- Model Size: ~9.3MB
- Format: PyTorch (.bin)
- Framework: Transformers
- Input: Plant leaf images
- Output: Disease classification with confidence scores