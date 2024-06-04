import torch
import os
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from trl import SFTTrainer

def setup():

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['MASTER_ADDR'] = '10.0.9.51'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ens5'
    dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=world_size, 
                        rank=rank)
    return rank, world_size
    

def cleanup():
    dist.destroy_process_group()

def print_gpu_status(local_rank):
    # 打印当前进程的GPU使用情况
    if torch.cuda.is_available():
        print(f"Process {os.getpid()} on local_rank {local_rank} is using GPU {local_rank}")
    else:
        print(f"No GPU available on local_rank {local_rank}, using CPU.")


def main():
    
    rank, world_size = setup()
    local_rank = rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model_id = "dickdiss/phi-3_qlora_consumer"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    
    )
     # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(model_id,
        quantization_config=bnb_config,
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto")

      # PeFT配置和模型适配器
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    model.add_adapter(lora_config, adapter_name="adapter")
    if hasattr(model, 'prepare_inputs_for_generation'):
        model.prepare_inputs_for_generation = model.prepare_inputs_for_generation

    # 应用DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    # 加载和预处理数据集
    data_file_path = "/home/ubuntu/lora/data.json"
    data = load_dataset("json", data_files=data_file_path, split='train').shuffle(seed=1234)
    data = data.map(lambda samples: tokenizer(samples["instruction"], padding=True, truncation=True), batched=True)
    
    
  
    # 设置训练参数和初始化训练器
    output_dir = "/home/ubuntu/outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=12,
        warmup_steps=1,
        max_steps=500,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    )
    train_sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=data, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
  

    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        args=args,
        train_dataset=train_loader,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        dataset_text_field="instruction"
        
    )

        # 执行训练
    trainer.train()
      
    

    # 推送模型和分词器到 Hugging Face Hub
    model.module.push_to_hub("dickdiss/phi-3_qlora_consumer")
    tokenizer.push_to_hub("dickdiss/phi-3_qlora_consumer")

    cleanup()

if __name__ == "__main__":
    main()

