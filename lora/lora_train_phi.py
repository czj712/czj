import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
import os
import json
import wandb

wandb.init(project="phi3_lora_finetuning")
def check_model_layers(model):
    layer_shapes = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_shapes[name] = module.weight.shape
    return layer_shapes

layer_shapes = check_model_layers(model)
for name, shape in layer_shapes.items():
    print(f"Layer: {name}, Shape: {shape}")

# 加载预训练的 phi-3 模型
model_id = "/home/u202220081001066/phi3"
model = AutoModelForCausalLM.from_pretrained(model_id,
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
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
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

# 加载数据集
data_file_path = "/users/u202220081001066/datas/single_review_rp_gpt_outputs.json"
data = load_dataset("json",data_files=data_file_path, split='train')
data = data.shuffle(seed=1234)  # 打乱数据集

data = data.map(lambda samples: tokenizer(samples["instruction"]), batched=True)
data = data.train_test_split(test_size=0.1)
train_data = data["train"]
test_data = data["test"]


# PeFT 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

output_dir = "/users/u202220081001066/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 初始化 Trainer
trainer = SFTTrainer(
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
        warmup_steps=1,
        max_steps=-1,
        learning_rate=4e-4,
        bf16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 训练模型
trainer.train()
# 保存和上传 Lora 适配器
adapter_output_dir = os.path.join(output_dir, "phi3_adapter")
if not os.path.exists(adapter_output_dir):
    os.makedirs(adapter_output_dir)

model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
wandb.finish()
