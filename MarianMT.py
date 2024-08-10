from transformers import MarianMTModel, MarianTokenizer

# MarianMT 모델과 토크나이저 로드
model_name = "Helsinki-NLP/opus-mt-en-ko"  # 영어에서 한국어로 번역
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 예시 텍스트 입력
input_text = "Hugging Face is a company that provides open-source tools to democratize AI."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 번역 생성
outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

# 결과 출력
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translated Text:", translated_text)
