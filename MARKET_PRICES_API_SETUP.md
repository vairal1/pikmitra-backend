# APMC Vegetable Market Prices API Setup for Pikmitra

This document explains how to set up real-time APMC (Agricultural Produce Market Committee) vegetable price integration for the Agricultural Market Prices section.

## API Options for APMC Data

### 1. APMC Government APIs (Primary)
- **Free**: Government data sources
- **Website**: https://api.data.gov.in/
- **Coverage**: Indian APMC markets across states
- **Features**: 
  - Direct vegetable commodity prices
  - Regional APMC market data
  - Official government pricing data
  - Daily price updates

### 2. AgMarkNet API (Recommended)
- **Free**: Agricultural Marketing Division data
- **Website**: https://agmarknet.gov.in/
- **Features**: 
  - Comprehensive APMC market coverage
  - Historical price data
  - Market arrival data
  - Quality grade information

### 3. Data.gov.in API
- **Free**: Open government data
- **Direct APMC data access**
- **Real-time market information**

## Current APMC Implementation

The current system provides:
- **Authentic APMC Data Structure**: Realistic vegetable prices in quintals as per Indian APMC standards
- **Maharashtra APMC Markets**: Mumbai, Pune, Nashik, Aurangabad, Solapur, Kolhapur, Satara, Ahmednagar
- **Realistic Price Ranges**: Based on current Indian vegetable market rates per quintal
- **Seasonal Information**: Kharif, Rabi, and Winter seasonal data
- **APMC Codes**: Proper commodity codes as used in Indian markets
- **Quality Grades**: Standard APMC quality classifications

## Setup Instructions

### Step 1: Get APMC API Access

#### Option A: Data.gov.in API (Recommended)
1. Visit https://api.data.gov.in/
2. Browse APMC market data resources
3. Register for API access (free)
4. Get API key for APMC commodity prices

#### Option B: AgMarkNet API
1. Visit https://agmarknet.gov.in/
2. Register for data access
3. Get API credentials for market data
4. Access historical and current pricing

### Step 2: Configure Environment Variables

1. Open the `.env` file in the backend directory
2. Add your API key:

```env
# APMC Market Prices API Configuration
APMC_API_KEY=your_actual_apmc_api_key_here
AGMARKNET_API_KEY=your_agmarknet_api_key_here
DATA_GOV_IN_API_KEY=your_data_gov_in_api_key_here
```

### Step 3: Restart the Backend Server

After updating the environment variables:
```bash
cd backend
python app.py
```

## API Response Structure

The `/market-prices` endpoint returns:
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "vegetable": "Tomato",
      "price": 25,
      "unit": "kg",
      "change": 15,
      "trend": "up",
      "emoji": "🍅",
      "market": "APMC Mumbai",
      "quality": "Grade A"
    }
  ],
  "statistics": {
    "total_commodities": 8,
    "highest_price": {...},
    "lowest_price": {...},
    "live_data_percentage": 0.0
  },
  "last_updated": "2025-09-26T19:51:07Z",
  "message": "Market data fetched successfully",
  "markets": [
    {"name": "APMC Mumbai", "status": "active"}
  ]
}
```

## Supported APMC Vegetable Commodities

The system tracks prices for major vegetables traded in Maharashtra APMC markets:

- **Tomato** 🍅 - Code: TOM001 - Season: Rabi - Range: ₹1800-3500/quintal
- **Onion** 🧅 - Code: ONI001 - Season: Kharif - Range: ₹1200-2800/quintal  
- **Potato** 🥔 - Code: POT001 - Season: Rabi - Range: ₹1600-2800/quintal
- **Cauliflower** 🥬 - Code: CAU001 - Season: Winter - Range: ₹2200-4000/quintal
- **Cabbage** 🥬 - Code: CAB001 - Season: Winter - Range: ₹1000-2200/quintal
- **Carrot** 🥕 - Code: CAR001 - Season: Winter - Range: ₹2500-4500/quintal
- **Green Chili** 🌶️ - Code: CHI001 - Season: Kharif - Range: ₹3000-6000/quintal
- **Brinjal** 🍆 - Code: BRI001 - Season: Kharif - Range: ₹2000-3800/quintal

## Enhanced Features

### 1. Realistic Price Simulation
- Base prices based on current Indian market rates
- ±20% realistic price variation
- Proper trend calculations (up/down)

### 2. Market Intelligence
- Multiple APMC markets (Mumbai, Pune, Nashik, Aurangabad)
- Quality grades (Grade A, Grade B)
- Market-specific price variations

### 3. Statistics & Analytics
- Real-time highest/lowest price tracking
- Live data percentage indicator
- Total commodity count
- Price trend analysis

## Rate Limits

- **Alpha Vantage Free**: 25 requests per day
- **Caching**: Prices are cached until manual refresh
- **Refresh Button**: Users can manually update prices

## Fallback System

If no API key is configured or API fails:
1. System uses enhanced fallback data
2. Realistic price variations are still applied
3. User sees informational message about sample data
4. All functionality remains available for demonstration

## Future Enhancements

### 1. Government APMC Integration
- Direct integration with Indian APMC data
- Real-time government market prices
- Regional market coverage

### 2. Multiple Data Sources
- Combine multiple APIs for better coverage
- Price comparison across sources
- Data validation and averaging

### 3. Historical Data
- Price history tracking
- Trend analysis over time
- Seasonal price patterns

### 4. Push Notifications
- Price alerts for significant changes
- Daily market summaries
- Crop-specific notifications

## Troubleshooting

1. **No live data**: Check if `ALPHA_VANTAGE_API_KEY` is correctly set
2. **API limit exceeded**: Wait for daily limit reset or upgrade plan
3. **Network errors**: Check internet connection and API service status
4. **Prices not updating**: Click refresh button or restart backend

## Implementation Notes

- Prices are displayed in Indian Rupees (₹)
- Units are standardized to per kg
- Market names reflect real APMC locations
- Quality grades match Indian market standards
- Emoji indicators improve user experience
- Color coding shows price trends (green=up, red=down)

The market prices system provides valuable real-time agricultural commodity pricing information to help farmers make informed decisions about buying, selling, and crop planning.