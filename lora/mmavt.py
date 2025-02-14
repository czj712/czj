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

# 初始化 WandB 项目
wandb.init(project="llama3_ramvat_finetuning")

# 定义模型路径
model_id = "/home/u202220081001066/llama3"

# 定义数据集路径
data_file_path = "/users/u202220081001066/datas/single_review_rp_gpt_outputs.json"

# 定义输出目录
output_dir = "llama3_mmavt_adapter"

# 定义训练参数
training_args = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": -1,
    "learning_rate": 4e-3,
    "bf16": True,
    "fp16": False,
    "logging_steps": 5,
    "output_dir": output_dir,
    "optim": "paged_adamw_8bit",
    "save_strategy": "epoch",
    "report_to": "wandb"
}

# 定义 Vera 配置
vera_configs = [
    {
        "svd_init": True,
        "lambda_lr_ratio": 16.0,
        "experiment_name": "svd_init_true"
    },
    {
        "svd_init": False,
        "lambda_lr_ratio": 16.0,
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

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 定义检查模型层的函数
def check_model_layers(model):
    layer_shapes = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_shapes[name] = module.weight.shape
    return layer_shapes

# 检查模型层
layer_shapes = check_model_layers(model)
for name, shape in layer_shapes.items():
    print(f"Layer: {name}, Shape: {shape}")

# 设置模型配置
model.config.use_cache = False

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载数据集
data = load_dataset("json", data_files=data_file_path)
data = data.shuffle(seed=123)

# 数据预处理函数
def preprocess_function(samples):
    return tokenizer(samples["instruction"], padding="max_length", truncation=True, max_length=512)

# 应用预处理函数
data = data.map(preprocess_function, batched=True)

# 分割数据集
split_data = data['train'].train_test_split(test_size=0.1)
train_data = split_data["train"]
test_data = split_data["test"]


# 进行对比实验
for config in vera_configs:
    # 初始化 Vera 配置
    vera_config = VeraConfig(
        target_modules=["q_proj", "o_proj"],
        r=64,
        vera_dropout=0.05,
        bias="none",
        svd_init=config["svd_init"],
        lambda_lr_ratio=config["lambda_lr_ratio"],
        d_initial=0.1
    )

    # 获取 PEFT 模型
    model = get_peft_model(model, vera_config)

    # 初始化 WandB 实验
    wandb.init(
        project="llama3_ramvat_finetuning",
        name=config["experiment_name"],
        config={
            "svd_init": config["svd_init"],
            "lambda_lr_ratio": config["lambda_lr_ratio"],
            "learning_rate": training_args["learning_rate"],
            "batch_size": training_args["per_device_train_batch_size"],
            "epochs": training_args["num_train_epochs"]
        }
    )

    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        dataset_text_field="instruction",
        peft_config=vera_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(**training_args),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 训练模型
    trainer.train()

    # 保存模型和分词器
    adapter_output_dir = os.path.join(output_dir, config["experiment_name"])
    if not os.path.exists(adapter_output_dir):
        os.makedirs(adapter_output_dir)
    trainer.model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)

    # 结束当前 WandB 实验
    wandb.finish()
