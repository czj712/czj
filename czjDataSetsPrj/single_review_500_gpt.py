import json
from openai import OpenAI

client = OpenAI(
    api_key="sk-proj-c3BbjBEQb5Xlqzff2tfJT3BlbkFJquzNSCeRh0wwLMOaTO4j",
)

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_user_profile(review, user_id):
    # 根据给定的商品信息和评论生成用户画像的提示
    prompt = f'''
According to consumer A(user_id:{user_id})'s comment information:\n  '{review}', What is the user portrait of consumer A?\n Just return a word. 
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
    data = load_data("/users/u202220081001066/datas/1comment_result.json")
    results = []
    print("开始处理数据...")

    # 为每一个用户生成用户画像，并保存到results列表中
    for user_data in data: 
        user_id = user_data[0]['user_id']
        review = user_data[0]['text']
        user_profile = generate_user_profile(review, user_id)
        result_dict = {
                    "instruction": f"According to consumer A(user_id:{user_id})'s comment information:\n  '{review}', What is the user portrait of consumer A?\n Just return a word. ",
                    "input": "",
                    "output": user_profile
                }
        results.append(result_dict)
        print(f"已处理用户 {user_id}。")
        
                    

    # 将结果保存到JSON文件中
    json_file_path = '/users/u202220081001066/datas/single_review_rp_gpt_outputs.json'
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {json_file_path} 文件中。")

if __name__ == "__main__":
    process_data()