"""
Test script for the Flask backend with PyTorch MobileNetV2
Tests all endpoints to ensure they're working correctly
"""

import requests
import json
import base64
import io
from PIL import Image
import numpy as np
import time

# Backend URL
BASE_URL = 'http://localhost:5000'

def create_test_image():
    """Create a test image for prediction"""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_news():
    """Test news endpoint"""
    print("\n🔍 Testing news endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/news")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ News endpoint working: {len(data.get('articles', []))} articles")
            return True
        else:
            print(f"❌ News endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ News endpoint error: {e}")
        return False

def test_diseases():
    """Test diseases endpoint"""
    print("\n🔍 Testing diseases endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/diseases")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Diseases endpoint working: {len(data.get('diseases', []))} diseases")
            return True
        else:
            print(f"❌ Diseases endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Diseases endpoint error: {e}")
        return False

def test_prediction():
    """Test prediction endpoint"""
    print("\n🔍 Testing prediction endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Test prediction
        payload = {
            "image": test_image,
            "language": "en"
        }
        
        response = requests.post(f"{BASE_URL}/predict-base64", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Prediction working: {data.get('disease')} (confidence: {data.get('confidence', 0):.2f})")
                return True
            else:
                print(f"❌ Prediction failed: {data.get('message')}")
                return False
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        return False

def test_disease_details():
    """Test disease details endpoint"""
    print("\n🔍 Testing disease details endpoint...")
    try:
        # First get diseases list
        response = requests.get(f"{BASE_URL}/diseases")
        if response.status_code == 200:
            data = response.json()
            diseases = data.get('diseases', [])
            if diseases:
                # Test first disease
                first_disease = diseases[0]
                disease_key = first_disease.get('key')
                
                # Get details
                response = requests.get(f"{BASE_URL}/disease/{disease_key}")
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"✅ Disease details working: {data.get('disease', {}).get('name')}")
                        return True
                    else:
                        print(f"❌ Disease details failed: {data.get('message')}")
                        return False
                else:
                    print(f"❌ Disease details endpoint failed: {response.status_code}")
                    return False
            else:
                print("❌ No diseases available for testing")
                return False
        else:
            print(f"❌ Could not fetch diseases list: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Disease details endpoint error: {e}")
        return False

def test_register():
    """Test user registration endpoint"""
    print("\n🔍 Testing user registration endpoint...")
    try:
        # Use a unique email for each test run
        unique_email = f"testuser_{int(time.time())}@example.com"
        payload = {
            "name": "Test User",
            "email": unique_email,
            "password": "password123"
        }
        response = requests.post(f"{BASE_URL}/register", json=payload)
        if response.status_code == 201:
            data = response.json()
            if data.get('success'):
                print(f"✅ User registration successful for {unique_email}")
                return True
            else:
                print(f"❌ Registration failed with success=false: {data.get('message')}")
                return False
        else:
            print(f"❌ Registration endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Registration endpoint error: {e}")
        return False

def test_login_and_profile():
    """Test user login and protected profile endpoint"""
    print("\n🔍 Testing user login and profile endpoints...")
    try:
        # 1. Register a user first
        unique_email = f"testuser_{int(time.time())}@example.com"
        password = "password123"
        register_payload = {"name": "Login Test User", "email": unique_email, "password": password}
        reg_response = requests.post(f"{BASE_URL}/register", json=register_payload)
        if reg_response.status_code != 201:
            print(f"❌ Prerequisite failed: Could not register user for login test.")
            return False
        
        # 2. Attempt to login
        login_payload = {"email": unique_email, "password": password}
        login_response = requests.post(f"{BASE_URL}/login", json=login_payload)
        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.status_code} - {login_response.text}")
            return False
        
        login_data = login_response.json()
        if not login_data.get('success') or not login_data.get('access_token'):
            print(f"❌ Login failed with success=false or no token: {login_data.get('message')}")
            return False
        
        print("✅ User login successful.")
        
        # 3. Access protected profile route with token
        access_token = login_data['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}
        profile_response = requests.get(f"{BASE_URL}/profile", headers=headers)
        
        if profile_response.status_code == 200:
            profile_data = profile_response.json()
            if profile_data.get('success') and profile_data.get('user', {}).get('email') == unique_email:
                print(f"✅ Protected profile endpoint working for {unique_email}")
                return True
        
        print(f"❌ Accessing protected profile failed: {profile_response.status_code} - {profile_response.text}")
        return False
    except Exception as e:
        print(f"❌ Login/Profile endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Pikmitra Backend Tests...")
    print("=" * 60)
    
    tests = [
        test_health,
        test_register,
        test_login_and_profile,
        test_news,
        test_diseases,
        test_prediction,
        test_disease_details,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the backend logs for details.")
    
    return passed == total

if __name__ == "__main__":
    main()
