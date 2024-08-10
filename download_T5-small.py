from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 올바른 모델 식별자 사용
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# 예시 텍스트 입력
input_text = "translate to german: Hugging Face is a company that provides open-source tools to democratize AI."

# 텍스트를 토큰화
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 요약 생성
outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

# 결과 출력
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Summary:", summary)
