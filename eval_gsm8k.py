import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, PeftConfig

base_model_id = "/home/u202220081001066/llama3"
peft_model_path = "/users/u202220081001066/czj/lora/LLaMA3_GRPO/checkpoint-1400"
test_data_path = "/home/u202220081001066/grade-school-math/grade_school_math/data/test.jsonl"
output_file = "llama3_gsm8k_eval_results.txt"
batch_size = 4 
max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

class GSM8KDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
            "problem_id": item.get("problem_id", idx)  # 保留原始ID
        }

test_dataset = GSM8KDataset(test_data_path)
print(f"Loaded {len(test_dataset)} test examples")

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token  # 确保设置pad token
base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto")

print("加载base_model完毕！")
print("正在加载peft_model...")
model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

merged_model = model.merge_and_unload()
print("合并模型完成！")

# 答案处理函数
def extract_answer(text):
    """严格匹配GSM8K的答案格式"""
    # 匹配最后出现的 \boxed{} 格式
    match = re.search(r"\\boxed{([\d.,]+)}", text)
    if match:
        return match.group(1).replace(",", "").strip()  # 处理千分位逗号
    
    # 如果无boxed格式，匹配最后一个数字
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers[-1] if numbers else None

def evaluate_gsm8k():
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        questions = batch["question"]
        true_answers = [extract_answer(a) for a in batch["answer"]]
            
        # 生成带CoT提示的问题
        """prompted_questions = [
                f"Solve this problem step by step: {q}\nLet's think step by step."
                for q in questions
            ]"""
            
            # 批量生成
        inputs = tokenizer(
                questions,
                #prompted_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
        outputs = merged_model.generate(
                **inputs,
                max_new_tokens=256,  # 控制生成长度
                temperature=0.1,     # 降低随机性
                do_sample=False      # 使用greedy decoding保证可重复性
            )
            
            # 解码并处理答案
        decoded_answers = tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )
            
            # 处理每个样本
        for pred, true in zip(decoded_answers, true_answers):
            pred_answer = extract_answer(pred)
            if pred_answer is not None and pred_answer == true:
                correct += 1
            total += 1
    # 计算准确率
    accuracy = correct / total
    
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
 
    
    return accuracy

# 运行评估
if __name__ == "__main__":
    evaluate_gsm8k()
