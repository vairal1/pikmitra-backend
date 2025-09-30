import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GNEWS_API_KEY')
print(f'API Key loaded: {api_key}')

if api_key and api_key != 'your_gnews_api_key_here':
    url = 'https://gnews.io/api/v4/search'
    params = {
        'q': 'farming OR agriculture',
        'lang': 'en',
        'country': 'in',
        'max': 3,
        'apikey': api_key
    }
    
    try:
        print('Making API request...')
        response = requests.get(url, params=params, timeout=10)
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f'Articles found: {len(articles)}')
            
            if articles:
                print('\nFirst article:')
                article = articles[0]
                print(f'Title: {article.get("title", "No title")}')
                print(f'Source: {article.get("source", {}).get("name", "No source")}')
                print(f'URL: {article.get("url", "No URL")}')
            else:
                print('No articles returned')
                print(f'Response: {data}')
        else:
            print(f'Error response: {response.text}')
            
    except Exception as e:
        print(f'Request failed: {e}')
else:
    print('No valid API key found')