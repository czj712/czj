import torch
from datasets import load_dataset
from peft import VeraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import transformers
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
import os
import json
import wandb
import torch.nn as nn


# 定义模型路径
model_id = "/home/u202220081001066/llama3"

# 定义数据集路径
data_file_path = "/home/u202220081001066/grade-school-math/grade_school_math/data/train.jsonl"

# 定义输出目录
output_dir = "llama3_mmavt_adapter_gsm8k"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载数据集
data = load_dataset("json", data_files=data_file_path)
data = data.shuffle(seed=123)

# 数据预处理函数
def preprocess_function(samples):
    text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
        "labels": tokenized["input_ids"][0].clone()
    }

# 应用预处理函数
data = data.map(
    preprocess_function,
    batched=False,  # 逐样本处理避免维度问题
    remove_columns=["question", "answer"])

# 分割数据集
split_data = data['train'].train_test_split(test_size=0.1)
train_data = split_data["train"]
test_data = split_data["test"]

# 定义训练参数
training_args = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": -1,
    "learning_rate": 4e-4,
    "bf16": True,
    "fp16": False,
    "logging_steps": 5,
    "output_dir": output_dir,
    "optim": "paged_adamw_8bit",
    "save_strategy": "epoch",
    "report_to": "wandb",
    "remove_unused_columns": False
}

# 定义 Vera 配置
vera_configs = [
    {
        "svd_init": True,
        "lambda_lr_ratio": 32.0,
        "experiment_name": "svd_init_true"
    },
    {
        "svd_init": False,
        "lambda_lr_ratio": 32.0,
        "experiment_name": "svd_init_false"
    },
    {
        "svd_init": True,
        "lambda_lr_ratio": 1.0,
        "experiment_name": "svd_init_true_no_double_lr"
    },
    {
        "svd_init": False,
        "lambda_lr_ratio": 1.0,
        "experiment_name": "svd_init_false_no_double_lr"
    }
]

# 进行对比实验
for config in vera_configs:
    # 释放之前模型占用的显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 初始化 WandB 实验（每次实验独立）
    wandb.init(
        project="llama3_mmavt_finetuning_gsm8k",
        name=config["experiment_name"],
        config={
            "svd_init": config["svd_init"],
            "lambda_lr_ratio": config["lambda_lr_ratio"],
            "learning_rate": training_args["learning_rate"],
            "batch_size": training_args["per_device_train_batch_size"],
            "epochs": training_args["num_train_epochs"]
        }
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 设置模型配置
    model.config.use_cache = False
    
    experiment_name = config["experiment_name"]
    svd_init = config["svd_init"]
    lambda_lr_ratio = config["lambda_lr_ratio"]
    print(f"\n开始实验：{experiment_name}")
    print(f"配置参数：svd_init={svd_init}, lambda_lr_ratio={lambda_lr_ratio}")


    # 初始化 Vera 配置
    vera_config = VeraConfig(
        target_modules=["q_proj", "o_proj"],
        r=128,
        vera_dropout=0.05,
        bias="none",
        svd_init=config["svd_init"],
        lambda_lr_ratio=config["lambda_lr_ratio"],
        d_initial=0.1
    )

    # 获取 PEFT 模型
    model = get_peft_model(model, vera_config)

    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=vera_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(**training_args),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    print(f"开始训练实验：{experiment_name}")
    # 训练模型
    trainer.train()

    # 保存模型和分词器
    adapter_output_dir = os.path.join(output_dir, config["experiment_name"])
    if not os.path.exists(adapter_output_dir):
        os.makedirs(adapter_output_dir)
    print(f"保存模型到：{adapter_output_dir}")
    trainer.model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)

    # 结束当前 WandB 实验
    wandb.finish()
    print(f"实验 {experiment_name} 完成！\n")
