import torch
from datasets import load_dataset
from peft import VeraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import transformers
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainingArguments
import os
import json
import wandb
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

# 定义模型路径
model_id = "/home/u202220081001066/llama3"

# 定义数据集路径
data_file_path = "/users/u202220081001066/datas/single_review_rp_gpt_outputs.json"

# 定义输出目录
output_dir = "llama3_mmavt_adapter"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载数据集
data = load_dataset("json", data_files=data_file_path)
data = data.shuffle(seed=123)

# 数据预处理函数
def preprocess_function(samples):
    text = f"Instruction: {samples['instruction']}\nInput: {samples['input']}\nOutput: {samples['output']}"
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
    remove_columns=["instruction", "input", "output"])

# 分割数据集
split_data = data['train'].train_test_split(test_size=0.1)
train_data = split_data["train"]
test_data = split_data["test"]


@dataclass
class MMAVTTrainingArguments(SFTConfig):
    lambda_lr_ratio: Optional[float] = field(
        default=16.0,
        metadata={"help": "lambda参数学习率比例 (lr_lambda_b = base_lr * ratio, lr_lambda_d = base_lr)"}
    )
    packing: bool = field(
        default=False,  # 添加缺失的packing参数
        metadata={"help": "Whether to use packing for SFTTrainer."}
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for dataset preprocessing."}
    )

# 自定义优化器创建函数
def create_mmavt_optimizer(model, optimizer_cls, optimizer_kwargs, lambda_lr_ratio):
    """
    创建支持 VeRA 双学习率的优化器
    """

    param_groups = {
        "base": {"params": [], "weight_decay": optimizer_kwargs.get("weight_decay", 0.0)},
        "lambda_b": {"params": [], "weight_decay": 0.0},  # 通常B参数不设权重衰减
        "lambda_d": {"params": [], "weight_decay": 0.0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "vera_lambda_b" in name:
            param_groups["lambda_b"]["params"].append(param)
        elif "vera_lambda_d" in name:
            param_groups["lambda_d"]["params"].append(param)
        else:
            # 处理无衰减参数
            param_groups["base"]["params"].append(param)

    base_lr = optimizer_kwargs["lr"]
    optimizer_grouped_parameters = [
        {
            "params": param_groups["base"]["params"],
            "weight_decay": param_groups["base"]["weight_decay"],
            "lr": base_lr
        },
        {
            "params": param_groups["lambda_b"]["params"],
            "weight_decay": param_groups["lambda_b"]["weight_decay"],
            "lr": base_lr * lambda_lr_ratio
        },
        {
            "params": param_groups["lambda_d"]["params"],
            "weight_decay": param_groups["lambda_d"]["weight_decay"],
            "lr": base_lr
        }
    ]

    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

# 自定义 Trainer 类
class MMAVTTrainer(SFTTrainer):
    def create_optimizer(self):
        # 获取优化器类和参数
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        
        # 创建自定义优化器
        self.optimizer = create_mmavt_optimizer(
            self.model,
            optimizer_cls,
            optimizer_kwargs,
            lambda_lr_ratio=self.args.lambda_lr_ratio
        )
        return self.optimizer


# 定义训练参数
training_args_base = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "learning_rate": 4e-4,  # 基础学习率
    "bf16": True,
    "logging_steps": 5,
    "output_dir": output_dir,
    "optim": "paged_adamw_8bit",
    "save_strategy": "epoch",
    "report_to": "wandb",
    "remove_unused_columns": False,
    "packing": False
}

# 定义 Vera 配置
vera_configs = [
    {
        "svd_init": True,
        "lambda_lr_ratio": 1.0,
        "experiment_name": "no_double_lr"
    },
    {
        "svd_init": True,
        "lambda_lr_ratio": 2.0,
        "experiment_name": "2_double_lr"
    },
    {
        "svd_init": True,
        "lambda_lr_ratio": 4.0,
        "experiment_name": "4_double_lr"
    },
    {
        "svd_init": True,
        "lambda_lr_ratio": 8.0,
        "experiment_name": "8_double_lr"
    }，
      {
        "svd_init": True,
        "lambda_lr_ratio": 16.0,
        "experiment_name": "16_double_lr"
    }，
    {
        "svd_init": True,
        "lambda_lr_ratio": 32.0,
        "experiment_name": "32_double_lr"
    }
]

# 进行对比实验
for config in vera_configs:
    # 释放之前模型占用的显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 初始化 WandB 实验（每次实验独立）
    wandb.init(
        project="llama3_mmavt_finetuning",
        name=config["experiment_name"],
        config={
            "svd_init": config["svd_init"],
            "lambda_lr_ratio": config["lambda_lr_ratio"],
            "base_lr": training_args_base["learning_rate"],
            "actual_lr_lambda_b": training_args_base["learning_rate"] * config["lambda_lr_ratio"],
            "actual_lr_lambda_d": training_args_base["learning_rate"],
            "batch_size": training_args_base["per_device_train_batch_size"],
            "epochs": training_args_base["num_train_epochs"]
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
        d_initial=0.1
    )

    # 获取 PEFT 模型
    model = get_peft_model(model, vera_config)

    training_args = MMAVTTrainingArguments(
        **training_args_base,
        lambda_lr_ratio=config["lambda_lr_ratio"]  # 注入比例参数
    )
    # 初始化训练器
    trainer = MMAVTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=vera_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
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
