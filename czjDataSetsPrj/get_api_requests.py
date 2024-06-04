import requests
import json
import pymysql

# MySQL数据库连接配置
config = {
    'host': 'asm5712-cluster.cluster-cmkbamxb6k6y.us-west-2.rds.amazonaws.com',
    'user': 'dev_czj',
    'password': 'dev_czj0511dev_czj',
    'database': 'consumer_agents',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 连接数据库
connection = pymysql.connect(**config)

# 从amazon_data_user_tag表中随机选择300条数据，拿到userid和category
try:
    with connection.cursor() as cursor:
        sql = "SELECT user_id, category FROM amazon_data_user_tag ORDER BY RAND() LIMIT 300"
        cursor.execute(sql)
        data = cursor.fetchall()
finally:
    connection.close()

api_key = "9d207bf0-10f5-4d8f-a479-22ff5aeffad1" 

results = []

# 遍历数据进行请求
for index, item in enumerate(data, 1):
    userid = item['user_id']
    category = item['category']

    print(f"Processing item {index}/{len(data)}...")
    print(f"Item data: userid={userid}, category={category}")

    headers = {'Authorization': f'Bearer {api_key}', 'accept': 'application/json', 'Content-Type': 'application/json'}
    payload = {
        "userid": userid,
        "category": category,
        "display_fields": "user_id,parent_asin,text" ,
        "max_chars_len": 500,
        "order_by_helpfulvote": "true",
        "max_len": 1
    }
    #打印请求体
    print(f"Request payload: {json.dumps(payload, indent=2)}")

    response = requests.post('http://52.43.5.161:8030/v1/review/get_review_by_userid', headers=headers, json=payload)
    result = response.json()
    
    # 将结果添加到列表中
    results.append(result)
    
    # 打印响应结果
    print(f"Response: {json.dumps(result, indent=2)}")
    
# 保存结果到JSON文件
with open('/home/ubuntu/1comment_result.json', 'w') as f:
    json.dump(results, f)

print("All data processed and saved to results.json.")
