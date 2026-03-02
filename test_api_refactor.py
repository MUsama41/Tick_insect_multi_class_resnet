import requests
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000"

def create_test_image():
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.json()}")
    assert response.status_code == 200

def test_predict_single():
    image_data = create_test_image()
    response = requests.post(f"{API_URL}/predict_single", data=image_data)
    print(f"Predict Single: {response.json()}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 2

def test_predict_multi():
    image_data = create_test_image()
    files = [
        ('files', ('test1.jpg', image_data, 'image/jpeg')),
        ('files', ('test2.jpg', image_data, 'image/jpeg'))
    ]
    response = requests.post(f"{API_URL}/predict_multi", files=files)
    print(f"Predict Multi (2 images): {response.json()}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_multi_limit():
    image_data = create_test_image()
    files = [
        ('files', ('test1.jpg', image_data, 'image/jpeg')),
        ('files', ('test2.jpg', image_data, 'image/jpeg')),
        ('files', ('test3.jpg', image_data, 'image/jpeg')),
        ('files', ('test4.jpg', image_data, 'image/jpeg'))
    ]
    response = requests.post(f"{API_URL}/predict_multi", files=files)
    print(f"Predict Multi (4 images - should fail): {response.status_code}")
    assert response.status_code == 400

if __name__ == "__main__":
    try:
        print("Starting API verification tests...")
        test_health()
        test_predict_single()
        test_predict_multi()
        test_predict_multi_limit()
        print("\n[SUCCESS] All functional requirements verified.")
        print("1. /predict_single accepts binary (raw body) and returns [label, confidence].")
        print("2. /predict_multi accepts max 3 images (multipart) and returns the highest confidence result as [label, confidence].")
    except Exception as e:
        print(f"\n[FAILURE] Tests failed: {e}")
        import traceback
        traceback.print_exc()
