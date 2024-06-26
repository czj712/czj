import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
import os
import json
import wandb

wandb.init(project="llama3_finetuning")

# 加载模型
model_id = "/home/u202220081001066/llama3"
model = AutoModelForCausalLM.from_pretrained(model_id,
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto")

model.config.use_cache=False
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id

# 加载数据集
data_file_path = "/users/u202220081001066/datas/single_review_rp_gpt_outputs.json"
data = load_dataset("json",data_files=data_file_path)
data = data.shuffle(seed=123)  # 打乱数据集
def preprocess_function(samples):
    return tokenizer(samples["instruction"], padding="max_length", truncation=True, max_length=512)

data = data.map(preprocess_function, batched=True)
# 分割数据集
split_data = data['train'].train_test_split(test_size=0.1)
train_data = split_data["train"]
test_data = split_data["test"]

# PeFT 配置
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

output_dir = "/users/u202220081001066/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def print_gpu_utilization(step):
    allocated = torch.cuda.memory_allocated(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    print(f"Step {step}: GPU memory allocated: {allocated / (1024 ** 3):.2f}GB, Max GPU memory allocated: {max_allocated / (1024 ** 3):.2f} GB")

class CustomTrainer(SFTTrainer):
        def training_step(self, *args, **kwargs):
                step = self.state.global_step
                result = super().training_step(*args, **kwargs)
                print_gpu_utilization(step)
                return result

# 初始化 Trainer
trainer = CustomTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="instruction",
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=transformers.TrainingArguments(
        num_train_epochs= 5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=-1,
        learning_rate=1e-4,
        bf16=True,
        fp16=False,
        logging_steps=10,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        report_to="wandb"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 训练模型
trainer.train()

# 保存和上传 Lora 适配器
adapter_output_dir = os.path.join(output_dir, "llama3_adapter")
if not os.path.exists(adapter_output_dir):
    os.makedirs(adapter_output_dir)

trainer.model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
wandb.finish()
