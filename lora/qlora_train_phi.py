import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
import os
import json

# 使用QLoRA量化加载模型
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload = False,
    llm_int8_skip_modules = None
    )
# 加载预训练的 phi-3 模型
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id,
    quantization_config=bnb_config,
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
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
data_file_path = "//home/ubuntu/czjDataSetsPrj/single_review_rp_gpt_outputs.json"
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
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=transformers.TrainingArguments(
        num_train_epochs= 10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=12,
        warmup_steps=1,
        max_steps=100,
        learning_rate=2e-4,
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

# 加载 LoRA 适配器并合并到模型中
adapter_path = adapter_output_dir  # 使用保存的适配器路径
model = PeftModel.from_pretrained(model, adapter_path)

# 将合并后的模型保存到本地
merged_output_dir = os.path.join(output_dir, "phi3_merged")
if not os.path.exists(merged_output_dir):
    os.makedirs(merged_output_dir)
model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)

model.push_to_hub("dickdiss/new_phi-3_qlora")
tokenizer.push_to_hub("dickdiss/new_phi-3_qlora")

