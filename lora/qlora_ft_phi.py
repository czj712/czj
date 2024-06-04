import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
# 加载模型和标记器
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_model_id = "/home/ubuntu/outputs/phi3_adapter"
merged_model_id = "/home/ubuntu/outputs/phi3_merged"
model = AutoModelForCausalLM.from_pretrained(merged_model_id,device_map={"":0}, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,)
#model = PeftModel.from_pretrained(model, adapter_model_id)
tokenizer = AutoTokenizer.from_pretrained(merged_model_id, padding_side='left', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
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
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
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