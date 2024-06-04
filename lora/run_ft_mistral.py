import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 定义量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型和标记器
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def get_completion(query: str, model, tokenizer) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造prompt模板
    prompt_template = """
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Question:
    {query}

    ### Answer:
    """
    prompt = prompt_template.format(query=query)

    # 使用tokenizer编码prompt文本
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to(device)

    # 使用模型生成文本
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
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
