import json
from openai import OpenAI

client = OpenAI(
    base_url="https://free.gpt.ge/v1",
    api_key="sk-0eb3qj6MQmUNXn632c0c499558354b4eB2243cF74dEe8a52",
)

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_user_profile(reviews, user_id):
    
    prompt = f"""
Based on the comments provided by user with ID {user_id}, generate a profile for the user. 
Comments: '{reviews}'
Return several descriptive tags separated by commas. Try to infer the age group and region if possible. If not, return other relevant tags like interests, buying habits, or sentiment."""
     
    try:
        response = client.chat.completions.create(
                 model="gpt-3.5-turbo",  # 使用适当的引擎
                 messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=60
                )
              
        user_profile = response.choices[0].message.content.strip()
        print(f"用户 {user_id} 的评论为: '{reviews}' 生成的用户画像为: {user_profile}")
        return user_profile
                 
    except Exception as e:
        print(f"用户 {user_id} 的评论为: '{reviews}' 生成用户画像时出错: {e}")
        return "Error generating profile"
def save_results(results, file_path):
    # 读取现有数据
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = []

    # 合并现有数据和新数据
    existing_data.extend(results)

    # 保存合并后的数据
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {file_path} 文件中。")

def process_data():
    count = 0
    save_interval = 50 
    data = load_data("/Users/zijianchen/Desktop/datas/5_merged_comments_data.json")
    json_file_path = '/Users/zijianchen/Desktop/datas/5_reviews_rp_gpt_outputs.json'
    results = []
    print("开始处理数据...")

    # 为每一个用户生成用户画像，并保存到results列表中
    for user_data in data:
        if user_data:  # 确保 user_data 不是空的
            user_id = user_data['user_id']
            reviews = user_data['comments']
            user_profile = generate_user_profile(reviews, user_id)
            result_dict = {
                "instruction": f"According to consumer A (user_id: {user_id})'s comments information: '{reviews}'. What is the user profile of consumer A? Just return several tags, separated by commas.",
                "input": "",
                "output": user_profile
            }
            results.append(result_dict)
            count += 1
            print(f"已处理用户 {user_id}。")
            
            if count % save_interval == 0:
                save_results(results, json_file_path)
                results.clear() 
        else:
            print("跳过空的 user_data。")

    if results:
        save_results(results, json_file_path)

       

if __name__ == "__main__":
    process_data()
