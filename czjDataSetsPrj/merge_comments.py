import json 
import os

existing_data_file="/users/u202220081001066/datas/5_merged_comments_data.json"
data_file = "/users/u202220081001066/datas/v1_5comment_result.json"
if os.path.exists(existing_data_file):
    with open(existing_data_file, 'r') as file:
        existing_data = json.load(file)
else:
    existing_data = []

with open(data_file, 'r') as file:
    data = json.load(file)

user_comments = []

# 遍历 data 中的每个用户的评论
for user_data in data:
    if not user_data:  # 检查 user_data 是否为空
        continue  # 如果为空，则跳过此条目

    # 创建一个空的字典，用于存储当前用户的数据
    user_info = {"user_id": user_data[0]["user_id"], "comments": []}
    # 遍历每条评论
    for comment in user_data:
        # 将每条评论的商品 ID 和评论文本添加到当前用户的评论列表中
        user_info["comments"].append(comment["text"])
        print(user_info)
    # 将当前用户的字典添加到用户评论列表中
    user_comments.append(user_info)

existing_data.extend(user_comments)
# 保存整理后的数据到文件
with open(existing_data_file, "w") as f:
    json.dump(user_comments, f)
print("文件已保存完成！")
