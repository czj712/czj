import requests
import json
import random

# 设置文件路径
file_paths = [
    '/users/u202220081001066/datas/file1.jsonl',
    '/users/u202220081001066/datas/file2.jsonl',
    '/users/u202220081001066/datas/file3.jsonl',
    '/users/u202220081001066/datas/file4.jsonl'
]

# 读取JSONL文件并提取数据
all_data = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        for line in file:
            all_data.append(json.loads(line))

# 随机选取500条数据
sampled_data = random.sample(all_data, 1000)

api_key = "9d207bf0-10f5-4d8f-a479-22ff5aeffad1"

results = []

# 遍历数据进行请求
for index, item in enumerate(sampled_data, 1):
    userid = item['user_id']
    category = item['category']

    print(f"Processing item {index}/{len(sampled_data)}...")
    print(f"Item data: userid={userid}, category={category}")

    headers = {'Authorization': f'Bearer {api_key}', 'accept': 'application/json', 'Content-Type': 'application/json'}
    payload = {
        "userid": userid,
        "category": category,
        "display_fields": "user_id,text",
        "max_chars_len": 500,
        "order_by_helpfulvote": "true",
        "max_len": 1
    }
    # 打印请求体
    print(f"Request payload: {json.dumps(payload, indent=2)}")

    response = requests.post('http://52.43.5.161:8030/v1/review/get_review_by_userid', headers=headers, json=payload)
    result = response.json()
    
    # 将结果添加到列表中
    results.append(result)
    
    # 打印响应结果
    print(f"Response: {json.dumps(result, indent=2)}")

# 保存结果到JSON文件
with open('/users/u202220081001066/datas/1comment_result.json', 'w') as f:
    json.dump(results, f)

print("All data processed and saved to results.json.")
