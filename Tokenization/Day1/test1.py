import re # 정규 표현식 라이브러리

def word_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # text에서 a~z와 0~9 그리고 공백을 제외한 모든것을 ""으로 바꾼다
    tokens = text.split() # 공백을 기준으로 리스트로 분리
    return tokens # 토큰 반환

print(word_tokenize("Hello World!"))
