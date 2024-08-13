from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 올바른 모델 식별자 사용
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# 예시 텍스트 입력
input_text = """translate to german: 
very lesson includes lots of hands-on exercises for you to try. Most of these are run in interactive notebooks, all of which are available on Kaggle. If you don’t work through the notebooks yourself, you’re not going to get nearly as much out of this course—so that means you need to get set up on Kaggle. We have a page to help you get going with Kaggle: click here to go there now. Instead of using Kaggle, another great option is Paperspace Gradient If you don’t have a Paperspace account yet, sign up with this link to get $10 credit (and we get a credit too).
"""


# 텍스트를 토큰화
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 요약 생성
outputs = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True)

# 결과 출력
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translate:", summary)
