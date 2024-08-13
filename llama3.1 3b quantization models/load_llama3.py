from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드
model_name = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LLM에게 보낼 메시지
input_text = "Hello, how are you today?"

# 입력 텍스트를 토큰화
inputs = tokenizer(input_text, return_tensors="pt")

# 모델 추론
outputs = model.generate(inputs["input_ids"], max_length=50)

# 모델 응답 디코딩
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
