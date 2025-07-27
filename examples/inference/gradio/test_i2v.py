#!/usr/bin/env python3
"""
Test script for T2V and I2V functionality in FastVideo Gradio app.
This script tests both the backend and frontend modifications.
"""

import requests
import json
import os
from PIL import Image
import numpy as np

def test_backend_t2v():
    """Test the backend T2V functionality directly"""
    backend_url = "http://localhost:8000"
    
    try:
        # Test T2V request data
        request_data = {
            "prompt": "A beautiful sunset over the ocean with gentle waves",
            "negative_prompt": "",
            "use_negative_prompt": False,
            "seed": 42,
            "guidance_scale": 7.5,
            "num_frames": 21,
            "height": 448,
            "width": 832,
            "num_inference_steps": 20,
            "randomize_seed": False,
            "return_frames": True,
            "image_path": None,
            "model_type": "t2v"
        }
        
        # Send request to backend
        response = requests.post(
            f"{backend_url}/generate_video",
            json=request_data,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend T2V test successful!")
            print(f"Success: {result.get('success')}")
            print(f"Seed used: {result.get('seed')}")
            if result.get('frames'):
                print(f"Frames returned: {len(result.get('frames'))}")
            else:
                print("No frames returned")
        else:
            print(f"❌ Backend T2V test failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Backend T2V test failed with exception: {e}")

def test_backend_i2v():
    """Test the backend I2V functionality directly"""
    backend_url = "http://localhost:8000"
    
    # Create a simple test image
    test_image = Image.new('RGB', (256, 256), color='red')
    temp_image_path = "test_image.png"
    test_image.save(temp_image_path)
    
    try:
        # Test I2V request data
        request_data = {
            "prompt": "The red square gently animates with subtle movement",
            "negative_prompt": "",
            "use_negative_prompt": False,
            "seed": 42,
            "guidance_scale": 7.5,
            "num_frames": 21,
            "height": 448,
            "width": 832,
            "num_inference_steps": 20,
            "randomize_seed": False,
            "return_frames": True,
            "image_path": temp_image_path,
            "model_type": "i2v"
        }
        
        # Send request to backend
        response = requests.post(
            f"{backend_url}/generate_video",
            json=request_data,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend I2V test successful!")
            print(f"Success: {result.get('success')}")
            print(f"Seed used: {result.get('seed')}")
            if result.get('frames'):
                print(f"Frames returned: {len(result.get('frames'))}")
            else:
                print("No frames returned")
        else:
            print(f"❌ Backend I2V test failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Backend I2V test failed with exception: {e}")
    
    finally:
        # Clean up test image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def test_backend_health():
    """Test if the backend is running"""
    backend_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing FastVideo T2V and I2V functionality...")
    print("=" * 50)
    
    # Test backend health first
    if test_backend_health():
        # Test T2V functionality
        print("\n📝 Testing T2V functionality...")
        test_backend_t2v()
        
        # Test I2V functionality
        print("\n🖼️  Testing I2V functionality...")
        test_backend_i2v()
    else:
        print("⚠️  Backend is not running. Please start the backend first.")
        print("You can start it with: python start_ray_serve_app.py")
    
    print("=" * 50)
    print("Test completed!") 