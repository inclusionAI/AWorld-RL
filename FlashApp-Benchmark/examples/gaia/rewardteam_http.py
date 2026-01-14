import requests
import json

# API endpoint
url = "https://agi.alipay.com/api/v1/reward/submit_task"

# Headers
headers = {
    "authorization": "sk-6d5d711540594b6ba1abaa79d36b684c",
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "model_name": "FlashRewardTeam",
    "query": "为地底探险游戏增加血量与道具功能",
    "artifact_id": "flashapp-976547fea4c54b7c-2",
    "render_url": "https://render.lingguangcontent.com/p/lingguang/2187ff5417657139325352910e1071/index.html"
}

# Send POST request
response = requests.post(url, headers=headers, json=payload)

# Check response
if response.status_code == 200:
    print("Request successful!")
    print("Response:", response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Response:", response.text)


# Get task result
task_id = "9aaab3c2-08d5-4cbc-b285-e6c836483dff"  # Replace with actual task_id from submit_task response
result_url = f"https://agi.alipay.com/api/v1/reward/{task_id}/task_result"

# Headers for GET request
result_headers = {
    "authorization": "sk-6d5d711540594b6ba1abaa79d36b684c",
    "Content-Type": "application/json"
}

# Send GET request
result_response = requests.get(result_url, headers=result_headers)

# Check response
if result_response.status_code == 200:
    print("Get task result successful!")
    print("Result:", result_response.json())
else:
    print(f"Get task result failed with status code: {result_response.status_code}")
    print("Response:", result_response.text)