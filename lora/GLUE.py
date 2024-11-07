import torch
from datasets import load_dataset
from peft import VeraConfig, get_peft_model, PeftModel
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
wandb.init(project="roBerta-base-glue")


glue_tasks = ["cola", "sst2", "stsb", "mnli", "qnli", "rte"]
results = {}

# 初始化结果字典
for task in glue_tasks:
    # 加载数据集
    data_files = {
        "train": f"/home/u202220081001066/glue_datas/{task}/train.tsv",
        "validation": f"/home/u202220081001066/glue_datas/{task}/dev.tsv"
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    model_path = "/home/u202220081001066/roberta-base/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def preprocess_function(examples):
        if task in ["mrpc", "qqp", "stsb", "mnli", "rte", "qnli"]:
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128)
        else:
            return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
    encoded_dataset = dataset.map(preprocess_function, batched = True)
    train_dataset = encoded_dataset['train']
    eval_dataset = encoded_dataset['validation']

# 根据任务标签数量加载模型（某些任务为二分类，其他任务可能为多分类）
    num_labels = 3 if task == "mnli" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)



# PeFT 配置
    vera_config = VeraConfig(
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    r=128,
    vera_dropout=0.05,
    bias="none",
    )
    model = get_peft_model(model, vera_config)
    model.print_trainable_parameters()


    output_dir = f"/users/u202220081001066/outputs/results/{task}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
# 初始化 Trainer
    def compute_metrics(p):
        predictions, labels = p
        preds = predictions.argmax(axis=-1)
        accuracy = (preds == labels).astype(float).mean()
        return {"accuracy": accuracy}
    trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    args=TrainingArguments(
        num_train_epochs= 3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=0.1,
        max_steps=-1,
        learning_rate=2e-5,
        bf16=True,
        fp16=False,
        logging_dir=f"/users/u202220081001066/outputs/logs/{task}",
        output_dir=output_dir,
        report_to="wandb"
    ))

    print(f"开始任务 {task} 的训练")
    trainer.train()

    # 评估
    print(f"评估任务 {task}")
    results[task] = trainer.evaluate()
    print(f"任务 {task} 的评估结果: {results[task]}")
print('所有任务的评估结果:', results)
wandb.finish()
