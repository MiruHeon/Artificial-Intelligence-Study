import re # 정규 표현식 라이브러리

def word_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # text에서 a~z와 0~9 그리고 공백을 제외한 모든것을 ""으로 바꾼다
    tokens = text.split() # 공백을 기준으로 리스트로 분리
    return tokens # 토큰 반환

re_token = word_tokenize("Hello World!")

vocab = {} # 사전 정의
idx = 1 # 사전 요소 카운트

for token in re_token: # 토큰화 된 리스트의 요소를 token에 담는다
    if token not in vocab: # 만약 사전에 token이 없다면
        vocab[token] = idx # 사전의 o번째 요소로 정의
        idx += 1

encoded = [vocab[token] for token in re_token] # 숫자로 토큰화

print("tokens:", re_token)
print("vocab:", vocab)
print("encoded:", encoded)
