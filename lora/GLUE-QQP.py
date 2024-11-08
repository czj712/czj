import os
import torch
from datasets import load_dataset
from peft import VeraConfig, get_peft_model, PeftModel
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
wandb.init(project="roBerta-base-glue-qqp")


task = "QQP"
data_files = {
        "train": f"/home/u202220081001066/glue_datas/{task}/train.tsv",
        "validation": f"/home/u202220081001066/glue_datas/{task}/dev.tsv"
    }
dataset = load_dataset("csv", data_files=data_files, delimiter="\t", column_names=["id", "qid1", "qid2", "question1", "question2", "is_duplicate"], header=None)
model_path = "/home/u202220081001066/roberta-base/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Columns in the dataset:", dataset["train"].column_names)
def preprocess_function(examples):
    question1 = examples["question1"]
    question2 = examples["question2"]
    inputs = tokenizer(
                question1, 
                question2, 
                padding="max_length", 
                truncation=True, 
                max_length=128
            )
    labels = []
    for label in examples["is_duplicate"]:
        try:
            labels.append(int(label))
        except ValueError:
            print(f"Skipping non-integer label: {label}")
            labels.append(0)  # Assign a default label (e.g., 0) or handle as appropriate
    
    inputs["labels"] = labels
    return inputs
encoded_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = encoded_dataset['train']
eval_dataset = encoded_dataset['validation']

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)



# PeFT 配置
vera_config = VeraConfig(
    target_modules=["q_proj","k_proj","v_proj","out_proj"],
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
    print(f"Accuracy:{accuracy}")
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
#trainer.train()

    # 评估
print(f"评估任务 {task}")
evaluation_results = trainer.evaluate()
print(f"任务 {task} 的评估结果:", evaluation_results)
wandb.finish()
