from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 올바른 모델 식별자 사용
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# 예시 텍스트 입력
input_text = """translate to german: 
very lesson includes lots of hands-on exercises for you to try. Most of these are run in interactive notebooks, all of which are available on Kaggle. If you don’t work through the notebooks yourself, you’re not going to get nearly as much out of this course—so that means you need to get set up on Kaggle. We have a page to help you get going with Kaggle: click here to go there now. Instead of using Kaggle, another great option is Paperspace Gradient If you don’t have a Paperspace account yet, sign up with this link to get $10 credit (and we get a credit too).

Once you’ve got your Kaggle account set up, you’ll need to get familiar with Jupyter Notebook, which is the platform we use for most of this course (and which most deep learning researchers and engineers use for their work). Jupyter is the most popular tool for doing data science in Python, for good reason. It is powerful, flexible, and easy to use. We think you will love it! Since the most important thing for learning deep learning is writing code and experimenting, it’s important that you have a great platform for experimenting with code. If you haven’t used it before, we’ve provided this to help you get started: Jupyter Notebook 101.

OK, now that you have your Kaggle account and know how to use Jupyter, you’re ready to open the notebook for this lesson: here it is. For every lesson, you can find links to all notebooks used in the Resources section of the lesson web page. For instance, for lesson 1, you’ll see that section immediately below this one.

As well as watching the video and working through the notebooks, you should also read the relevent chapter(s) of the fast.ai book, Practical Deep Learning for Coders. Each lesson will tell you what chapter you need to read, just below the video. For this lesson, it’s chapter 1. There’s a few ways to read the book – you can buy it as a paper book or Kindle ebook, or you can read it for free as a Jupyter notebook. The whole book is written as Jupyter notebooks, so you can also execute all the code in the book yourself. To go to the interactive Jupyter version of any chapter, click The book in the left sidebar, where you’ll find a list of chapter links. You’ll also find links to read-only versions of each chapter there.
"""


# 텍스트를 토큰화
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 요약 생성
outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

# 결과 출력
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translate:", summary)
