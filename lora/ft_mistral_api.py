from fastapi import FastAPI, HTTPException, status
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 初始化FastAPI应用
app = FastAPI()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config, device_map={"":0})
tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

ft_mistral_model = AutoModelForCausalLM.from_pretrained("dickdiss/mistral_lora_consumer", quantization_config=bnb_config, device_map={"":0})
tokenizer_ft_tokenizer = AutoTokenizer.from_pretrained("dickdiss/mistral_lora_consumer")
def get_completion(query: str, model_name: str):
    # 选择要使用的模型和tokenizer
    if model_name == "mistral":
        # 加载mistral模型和tokenizer
        model = mistral_model
        tokenizer = tokenizer_mistral
    elif model_name == "ft_mistral":
        # 加载微调后的FT_Mistral模型和tokenizer
        model = ft_mistral_model
        tokenizer = tokenizer_ft_tokenizer
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model name")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造prompt模板
    prompt_template = """
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

# 定义API端点，接受POST请求，并根据输入数据返回模型生成的答案
@app.post("/predict/mistral")
async def predict(input_data: str):
    try:
        prediction = get_completion(input_data, "mistral")
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}, status.HTTP_500_INTERNAL_SERVER_ERROR
@app.post("/predict/ft_mistral")
async def predict(input_data: str):
    try:
        prediction = get_completion(input_data, "ft_mistral")
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}, status.HTTP_500_INTERNAL_SERVER_ERROR

# 运行FastAPI应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
