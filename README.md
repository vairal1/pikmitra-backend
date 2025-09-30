# Pikmitra Python Backend

A clean Flask-based backend for the Pikmitra agriculture app with PyTorch MobileNetV2 integration for plant disease detection.

## Features

- 🌱 **Plant Disease Detection**: Upload images and get AI-powered disease diagnosis using PyTorch MobileNetV2
- 📰 **Live News Integration**: Fetch latest agricultural news from NewsAPI
- 🌐 **Multilingual Support**: English, Hindi, and Marathi language support
- 📱 **Camera Integration**: Support for both file upload and camera capture
- 🔒 **CORS Enabled**: Ready for frontend integration
- 🤖 **PyTorch MobileNetV2**: Uses your trained PyTorch model

## Quick Start

### 1. Setup and Installation

```bash
# Run the automated setup script
start.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 2. Environment Variables

Create a `.env` file or set environment variables:

```env
FLASK_ENV=development
NEWS_API_KEY=your_news_api_key_here
PORT=5000
JWT_SECRET_KEY=please_change_me
DATABASE_URL=sqlite:///absolute/or/relative/path/to/database.db
```

If `DATABASE_URL` is not provided, a SQLite file `database.db` will be created inside `backend/`.

### 3. Testing

```bash
# Test all endpoints
python test.py

# Or test individual endpoints:
curl http://localhost:5000/health
curl http://localhost:5000/news
curl http://localhost:5000/diseases
```

## API Endpoints

### Health Check
- **GET** `/health` - Check backend status

### Plant Disease Detection
- **POST** `/predict` - Upload image file for disease prediction
- **POST** `/predict-base64` - Send base64 encoded image for prediction

### News
- **GET** `/news` - Fetch latest agricultural news

### Disease Information
- **GET** `/diseases` - Get list of all available diseases
- **GET** `/disease/<disease_key>` - Get detailed information about a specific disease

## PyTorch MobileNetV2 Integration

### Model Location

The backend automatically looks for your PyTorch MobileNetV2 model at:
```
../model/mobilenet_v2_1.0_224-plant-disease-identification/pytorch_model.bin
```

### Model Requirements

- **Format**: PyTorch model file (`.bin`)
- **Input**: RGB images resized to 224x224 pixels
- **Output**: Disease classification with confidence scores
- **Classes**: Should match the diseases in `cures.json`

### Model Loading

The backend automatically:
1. Loads your PyTorch model from the specified path
2. Loads disease class mappings from `config.json`
3. Uses GPU if available, otherwise CPU
4. Applies proper preprocessing for MobileNetV2

## Request/Response Examples

### Disease Prediction

**Request:**
```json
POST /predict-base64
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "language": "en"
}
```

**Response:**
```json
{
  "success": true,
  "disease": "Tomato - Late Blight",
  "disease_key": "Tomato___Late_blight",
  "confidence": 0.85,
  "cure": "Use fungicides like Mancozeb or Metalaxyl, remove infected plants, and improve drainage.",
  "language": "en"
}
```

### News

**Request:**
```bash
GET /news
```

**Response:**
```json
{
  "success": true,
  "articles": [
    {
      "title": "Latest Agricultural Technology Trends",
      "description": "New innovations in farming technology...",
      "url": "https://example.com/news1",
      "publishedAt": "2024-01-15T10:00:00Z",
      "source": {"name": "Agricultural News"}
    }
  ],
  "total": 20
}
```

## Frontend Integration

### React/TypeScript Example

```typescript
// Upload image and get prediction
const predictDisease = async (imageFile: File) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('language', 'en');

  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();
  return result;
};

// Get news
const fetchNews = async () => {
  const response = await fetch('http://localhost:5000/news');
  const data = await response.json();
  return data.articles;
};
```

### Camera Integration

```typescript
// Capture photo from camera
const capturePhoto = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ 
    video: { facingMode: 'environment' } 
  });
  
  // Capture image and convert to base64
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  // ... capture logic
  
  const imageDataUrl = canvas.toDataURL('image/jpeg');
  
  // Send to backend
  const response = await fetch('http://localhost:5000/predict-base64', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageDataUrl, language: 'en' })
  });
};
```

## File Structure

```
backend/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── cures.json               # Disease cure information
├── start.bat                # Windows startup script
├── test.py                  # Test script
├── README.md                # This file
└── uploads/                 # Upload directory (auto-created)
```

## Dependencies

- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities
- **Pillow**: Image processing
- **NumPy**: Numerical operations
- **Requests**: HTTP requests

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure your PyTorch model is at the correct path
2. **CORS errors**: Flask-CORS is enabled, check frontend URL
3. **News API failing**: Set `NEWS_API_KEY` environment variable
4. **Camera not working**: Check browser permissions and HTTPS requirement

### Debug Mode

```bash
set FLASK_ENV=development
python app.py
```

### Logs

The application logs important events:
- Model loading status
- Device information (CPU/GPU)
- Cures data loading
- API request/response details
- Error messages

## Production Deployment

### Environment Setup

```bash
set FLASK_ENV=production
set NEWS_API_KEY=your_production_api_key
set PORT=5000
```

### Security Considerations

- Use HTTPS in production
- Validate all input data
- Implement rate limiting
- Secure API keys
- Use production WSGI server (gunicorn)

## Support

For issues or questions:
1. Check the logs for error messages
2. Run the test script to verify functionality
3. Ensure all dependencies are installed
4. Verify your PyTorch model is at the correct path

## License

This project is part of the Pikmitra agriculture application.
