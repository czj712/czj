import pymysql
import pandas as pd
from openai import OpenAI
import json

# 数据库连接配置
config = {
    'host': 'asm5712-cluster.cluster-cmkbamxb6k6y.us-west-2.rds.amazonaws.com',
    'user': 'dev_czj',
    'password': 'dev_czj0511dev_czj',
    'database': 'consumer_agents',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 初始化OpenAI库
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-proj-mdmzPQj95v6veal6O6SOT3BlbkFJuAOqFANXfhXZaqVNQs0R",
)

def fetch_data():
    # 连接到MySQL数据库
    connection = pymysql.connect(**config)
    print("数据库连接成功。")
    try:
        with connection.cursor() as cursor:
            # 查询语句
            sql = "SELECT CONCAT(title, '; brand: ', brand, '; price: ', price) AS product_info, parent_asin AS product_id, reviews FROM amazon_data_czjtest"
            cursor.execute(sql)
            data = cursor.fetchall()
            print(f"查询到 {len(data)} 条数据。")
            return data
    finally:
        connection.close()
        print("数据库连接已关闭。")

def generate_user_profile(product_info, user_id, review):
    # 根据给定的商品信息和评论生成用户画像的提示
    prompt = f'''
Product information: {product_info}
Consumer A(user_id:{user_id})'s comment information: {review}
# Question: What is the user profile of consumer A?
# rules:
1、Do not use specific product names, abstract and macroscopic are required
2、Just return a word without punctuation or other symbols
3、Please answer in English
'''
    response = client.chat.completions.create(
        model="gpt-4",  # 使用适当的引擎
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=60
    )
    print(f"生成的用户画像为 {user_id}: {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

def process_data():
    data = fetch_data()
    results = []
    print("开始处理数据...")

    # 为每一条数据生成用户画像，并保存到results列表中
    for index, item in enumerate(data):
        product_info = item['product_info'] 
        product_id = item['product_id']
        reviews = json.loads(item['reviews'])
        for review in reviews:
            comment = review['text']
            user_id = review['user_id']
            user_profile = generate_user_profile(product_info, user_id, comment)
            result_dict = {
                "instruction": f"Consumer [{user_id}]'s comment on product '{product_id}' is: '{comment}' /n What is the user profile of consumer'{user_id}'?  Just return a word without punctuation or other symbols." ,
                "input": "",
                "output": user_profile
            }
            results.append(result_dict)
        print(f"已处理 {index + 1}/{len(data)} 条数据。")
                    

    # 将结果保存到JSON文件中
    json_file_path = 'rp_gpt_outputs.json'
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {json_file_path} 文件中。")


if __name__ == "__main__":
    process_data()
