import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. 취약점 분석
response = requests.get(f"{BASE_URL}/analysis/weakness", params={"user_id": "2"})
print("취약점 분석:", response.json())

# 2. 취약점 기반 추천
response = requests.get(f"{BASE_URL}/recommend/weakness", params={
    "user_id": "2",
    "k": 3
})
print("추천 결과:", response.json())
# 3. 배치 추천
response = requests.post(
    f"{BASE_URL}/recommend/batch?strategy=weakness",
    json={"user_ids": ["1", "2"], "k": 2}
)
print("배치 추천:", response.json())