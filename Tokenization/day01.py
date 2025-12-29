import re # 정규 표현식 라이브러리

vocab = {} # 사전 정의
idx = 1 # 사전 요소 카운트

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # text에서 a~z와 0~9 그리고 공백을 제외한 모든것을 ""으로 바꾼다
    tokens = text.split() # 공백을 기준으로 리스트로 분리
    return tokens

def add_vocab(tokens):
    global idx
    for token in tokens: # 토큰화 된 리스트의 요소를 token에 담는다
        if token not in vocab: # 만약 사전에 token이 없다면
            vocab[token] = idx # 사전의 o번째 요소로 정의
            idx += 1

def encoding(tokens):
    encoded = []
    not_in = []
    for token in tokens:
        if token in vocab: # 만약 사전에 토큰화 된 단어가 있는가?
            encoded.append(vocab[token]) # 그렇다면 숫자로 인코딩하라
        else:
            not_in.append(token)
            print(f"\"{token}\" is not token!")
    return encoded
    
# TEST 1
token_add = tokenize("Hello World!")
add_vocab(token_add)
print(encoding(token_add))

# TEST 2
token_add = tokenize("No World!")
print(encoding(token_add))
