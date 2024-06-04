import json 

with open("/home/ubuntu/czjDataSetsPrj/results.json") as file:
    data = json.load(file)

user_comments = []

# 遍历 data 中的每个用户的评论
for user_data in data:
    # 创建一个空的字典，用于存储当前用户的数据
    user_info = {"user_id": user_data[0]["user_id"], "comments": []}
    # 遍历每条评论
    for comment in user_data:
        # 将每条评论的商品 ID 和评论文本添加到当前用户的评论列表中
        user_info["comments"].append({
            "product_id": comment["parent_asin"],
            "review": comment["text"]
        })
    # 将当前用户的字典添加到用户评论列表中
    user_comments.append(user_info)

# 保存整理后的数据到文件
with open("/home/ubuntu/formated_data.json", "w") as f:
    json.dump(user_comments, f)
print("文件已保存完成！")
