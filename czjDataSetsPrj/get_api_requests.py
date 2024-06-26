import requests
import json
import random
import os

# 设置文件路径和对应的类别
file_paths = {
    '/users/u202220081001066/datas/Video_Games.jsonl': 'Video_Games',
    '/users/u202220081001066/datas/Musical_Instruments.jsonl': 'Musical_Instruments',
    '/users/u202220081001066/datas/Software.jsonl': 'Software' , 
    '/users/u202220081001066/datas/Industrial_and_Scientific.jsonl': 'Industrial_and_Scientific',
    '/users/u202220081001066/datas/Arts_Crafts_and_Sewing.jsonl': 'Arts_Crafts_and_Sewing' 
}

# 读取JSONL文件并提取数据，同时添加category字段
all_data = []
for file_path, category in file_paths.items():
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data['category'] = category
            all_data.append(data)

# 随机选取500条数据
sampled_data = random.sample(all_data, 500)

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
        "max_chars_len": 800,
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
with open('/users/u202220081001066/datas/v3_1comment_result.json', 'w') as f:
    json.dump(results, f)

print("All data processed and saved to v3_1comment_result.json.")
