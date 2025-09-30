# News API Setup for Pikmitra

This document explains how to set up news API integration for the Agriculture News section.

## API Options

### 1. GNews API (Recommended)
- **Free Tier**: 100 requests/day
- **Website**: https://gnews.io/
- **Features**: 
  - Keyword search
  - Country filtering (supports India)
  - Language filtering
  - Source filtering

### 2. NewsAPI (Alternative)
- **Free Tier**: 1000 requests/month
- **Website**: https://newsapi.org/
- **Features**: 
  - Comprehensive news coverage
  - Category filtering
  - Source filtering
  - More request limits for paid plans

## Setup Instructions

### Step 1: Get API Key

#### For GNews API:
1. Visit https://gnews.io/
2. Sign up for a free account
3. Copy your API key from the dashboard

#### For NewsAPI:
1. Visit https://newsapi.org/
2. Sign up for a free account
3. Copy your API key from the dashboard

### Step 2: Configure Environment Variables

1. Open the `.env` file in the backend directory
2. Add your API key:

```env
# For GNews API (recommended)
GNEWS_API_KEY=your_actual_api_key_here

# OR for NewsAPI
NEWS_API_KEY=your_actual_api_key_here
```

### Step 3: Restart the Backend Server

After updating the environment variables:
```bash
cd backend
python app.py
```

## News Categories

The system fetches news for two categories:

### 1. Farming News
**Keywords**: farming, agriculture, crop, harvest, irrigation, agri-tech, agricultural technology, smart farming

### 2. Government News  
**Keywords**: government scheme, subsidy, agriculture policy, agriculture ministry, farmer welfare, agricultural subsidy, farm loan, crop insurance

## Fallback Data

If no API key is configured or the API is unavailable, the system will show sample news articles to demonstrate the functionality.

## API Rate Limits

- **GNews Free**: 100 requests/day
- **NewsAPI Free**: 1000 requests/month

The application caches news data and only fetches new data when the "Refresh" button is clicked.

## Troubleshooting

1. **No news displayed**: Check if API key is correctly set in `.env` file
2. **API limit exceeded**: Wait for the limit to reset or upgrade to a paid plan
3. **Network errors**: Check internet connection and API service status
4. **CORS issues**: The backend handles CORS, no frontend configuration needed

## Implementation Details

The backend endpoint `/news` returns:
```json
{
  "success": true,
  "categories": {
    "farming": {
      "title": "Farming News",
      "articles": [...],
      "count": 3
    },
    "government": {
      "title": "Government News", 
      "articles": [...],
      "count": 3
    }
  },
  "total": 6,
  "message": "News fetched successfully"
}
```

Each article contains:
- `title`: Article headline
- `description`: Article summary
- `url`: Link to original article
- `publishedAt`: Publication timestamp
- `source.name`: News source name
- `image`: Article image URL (when available)