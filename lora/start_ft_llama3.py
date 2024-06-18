import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 加载模型和标记器
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_model_id = "/home/ubuntu/outputs/llama3_adapter"
merged_model_id = "/users/u202220081001066/outputs/llama3_merged"
model = AutoModelForCausalLM.from_pretrained(merged_model_id,device_map={"":0}, trust_remote_code=True, torch_dtype=torch.float16,)
#model = PeftModel.from_pretrained(model, adapter_model_id)
tokenizer = AutoTokenizer.from_pretrained(merged_model_id,  padding_side='right', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def get_completion(query: str, model, tokenizer) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造prompt模板
    prompt_template = f"""
    {query}

    Answer:
    """
    prompt = prompt_template.format(query=query) 

    # 使用tokenizer编码prompt文本
    encodeds = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=True)
    model_inputs = encodeds.to(device)

    # 使用模型生成文本
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256,repetition_penalty=1.5, pad_token_id=tokenizer.pad_token_id, num_return_sequences=1)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

# 交互式输入指令并生成答案
while True:
    query = input("Enter your question (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    else:
        answer = get_completion(query, model, tokenizer)
        print("Generated Answer:", answer)
