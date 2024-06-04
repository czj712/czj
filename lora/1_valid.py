import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
import os
import json

# 使用QLoRA量化加载模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    
)
# 加载预训练的 Mistral-7B 模型
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_id,
    quantization_config=bnb_config,
    device_map="auto")
try:
    with open(model_id, 'r') as f:
        index = json.loads(f.read())
except json.decoder.JSONDecodeError:
    print("Error: The file either contains no data or is not a valid JSON file.")
except FileNotFoundError:
    print("Error: The file was not found.")
model.config.use_cache=False
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id,  add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌为 EOS 令牌

# 加载数据集
data_file_path = "/home/ubuntu/lora/data.json"
data = load_dataset("json",data_files=data_file_path, split='train')
data = data.shuffle(seed=1234)  # 打乱数据集

data = data.map(lambda samples: tokenizer(samples["instruction"]), batched=True)
data = data.train_test_split(test_size=0.1)
train_data = data["train"]
test_data = data["test"]


# PeFT 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 准备模型进行 k 位训练
model = prepare_model_for_kbit_training(model)

# 添加 Lora 适配器
model.add_adapter(lora_config, adapter_name="adapter")

output_dir = "/home/ubuntu/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 初始化 Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="instruction",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 训练模型
trainer.train()


