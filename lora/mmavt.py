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
wandb.init(project="llama3_ramvat_finetuning")

# 加载模型
model_id = "/home/u202220081001066/llama3"
model = AutoModelForCausalLM.from_pretrained(model_id,
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto")

def check_model_layers(model):
    layer_shapes = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_shapes[name] = module.weight.shape
    return layer_shapes

layer_shapes = check_model_layers(model)
for name, shape in layer_shapes.items():
    print(f"Layer: {name}, Shape: {shape}")

model.config.use_cache=False
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

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
vera_config = VeraConfig(
    target_modules=["q_proj", "o_proj"],
    r=64,
    vera_dropout=0.05,
    bias="none",
    svd_init=True,
    lambda_lr_ratio=16.0,
    d_initial=0.1,
    )
model = get_peft_model(model, vera_config)
print(model.vera_A["default"][:5, :5])


output_dir = "/users/u202220081001066/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def get_parameter_groups(model, base_lr, lambda_lr_ratio):
    lambda_d_params = []
    lambda_b_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "vera_lambda_d" in name:
                lambda_d_params.append(param)
            elif "vera_lambda_b" in name:
                lambda_b_params.append(param)
            else:
                other_params.append(param)
    return [
        {'params': lambda_d_params, 'lr': base_lr},
        {'params': lambda_b_params, 'lr': base_lr * lambda_lr_ratio},
        {'params': other_params, 'lr': base_lr},
    ]
        

class CustomTrainer(SFTTrainer):
    def __init__(self, *args, peft_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.peft_config = peft_config
    def create_optimizer(self):
        if self.optimizer is None:
            base_lr = self.args.learning_rate
            lambda_lr_ratio = self.peft_config.lambda_lr_ratio
            parameter_groups = get_parameter_groups(self.model, base_lr, lambda_lr_ratio)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(parameter_groups, **optimizer_kwargs)
        return self.optimizer
    def training_step(self, model, inputs):
        step = self.state.global_step
        result = super().training_step(model, inputs)
        return result    


# 初始化 Trainer
base_lr = 4e-3
trainer = CustomTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="instruction",
    peft_config=vera_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=transformers.TrainingArguments(
        num_train_epochs= 5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=0.1,
        max_steps=-1,
        learning_rate=4e-3,
        bf16=True,
        fp16=False,
        logging_steps=5,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        report_to="wandb"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 训练模型
optimizer = trainer.create_optimizer()
for group in optimizer.param_groups:
    print(f"LR: {group['lr']}, Params: {len(group['params'])}")
trainer.train()

# 保存和上传 Lora 适配器
adapter_output_dir = os.path.join(output_dir, "llama3_mmavt_adapter")
if not os.path.exists(adapter_output_dir):
    os.makedirs(adapter_output_dir)

trainer.model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
wandb.finish()
