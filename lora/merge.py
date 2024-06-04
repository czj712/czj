import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 配置路径
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_path = "/home/ubuntu/outputs/llama3_adapter"
merged_output_dir = "/home/ubuntu/outputs/llama3_merged"

# 加载原始模型
print(f"Loading the base model from {model_id}")
base_model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False, torch_dtype=torch.float16, trust_remote_code=True)

# 加载LoRA适配器并合并到模型中
print(f"Loading the LoRA adapter from {lora_path}")
lora_model = PeftModel.from_pretrained(base_model, lora_path)

# 应用 LoRA 适配器
print("Applying the LoRA adapter")
model = lora_model.merge_and_unload()

# 保存合并后的模型
if not os.path.exists(merged_output_dir):
    os.makedirs(merged_output_dir)
model.save_pretrained(merged_output_dir)

# 加载和保存tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id,  padding_side='left', trust_remote_code=True)
tokenizer.save_pretrained(merged_output_dir)

model.push_to_hub("dickdiss/llama3_qlora_merged")
tokenizer.push_to_hub("dickdiss/llama3_qlora_merged")

