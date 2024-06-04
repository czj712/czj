import os
import torch 
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import deepspeed
local_env = os.environ.copy()
local_env["PATH"]="/home/ubuntu/anaconda3/envs/py310/bin" + local_env["PATH"]
os.environ.update(local_env)
deepspeed.ops.op_builder.CPUAdamBuilder().load()
# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)

# LoRA 和训练配置参数
model_id = "dickdiss/phi-3_qlora_consumer"
output_dir = "/home/ubuntu/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logging_dir = "/home/ubuntu/logs"
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
data_file_path = "/home/ubuntu/lora/data.json"
batch_size = 4
learning_rate = 2e-4
num_train_epochs = 5
gradient_accumulation_steps = 12
max_length = 2048
ds_config = "/home/ubuntu/lora/ds_config.json"

# 设置量化配置
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
)
checkpoint_path = "/home/ubuntu/outputs/checkpoint-20"
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_id,
    quantization_config=quantization_config,
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Tokenizer 和数据集
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

data = load_dataset("json", data_files=data_file_path, split='train').shuffle(seed=1234)
data = data.map(lambda samples: tokenizer(samples["instruction"], truncation=True, max_length=max_length), batched=True)
data = data.train_test_split(test_size=0.1)
train_data = data['train']
test_data = data['test']

# 数据 Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 应用 LoRA 适配器
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    deepspeed=ds_config,
    logging_dir=logging_dir,
    logging_steps=10,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    resume_from_checkpoint=False,
    save_on_each_node=True,
    save_only_model=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
model.push_to_hub("dickdiss/phi-3_qlora_consumer")
tokenizer.push_to_hub("dickdiss/phi-3_qlora_consumer")
