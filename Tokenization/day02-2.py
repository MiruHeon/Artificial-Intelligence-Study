# python -m venv venv     [venv : 환경 맞추기 (docker)같은 존재]
# .\venv\Scripts\activate [venv 활성화]
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process  [venv 활성화 명령어 실행 권한 부여]

# pip install tensorflow [텐서플로 라이브러리]
# keras : 딥 러닝 모델을 빌드하고 학습시키기 위한 Tensorflow의 상위 수준 API

# 토큰화 모듈 [문장 나눠서 단어사전 제작]
from tensorflow.keras.preprocessing.text import Tokenizer

# 패딩 모듈 [형식 맞추기]
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 임베딩 모듈 [AI가 알아듣는 규칙 만들기]
from tensorflow.keras.layers import Embedding

# ______________________________________________________________________

# ① 토큰화
tokenizer = Tokenizer( # 토큰화 객체 생성
    num_words = 1000, # 사전이 저장할 단어 개수
    oov_token="<OOV>" # 정의되지 않은 단어
)

texts = [ # 사전에 넣을 단어들 목록
    "Hello World",
    "Hello AI",
    "AI is future"
]

tokenizer.fit_on_texts(texts) # 공백을 기준으로 분류해서 사전에 넣는 작업

seq = tokenizer.texts_to_sequences(["Hello World AI"]) # 넘버링 토큰화
print(seq) # 단어 순서는 등장 횟수 순이다. 단 사전의 1번 요소는 <OOV>이다. 이때 등장 횟수가 같으면 먼저 나온 순으로 정렬된다.

# ② 패딩
max_len = 5 # 임베딩이 받을 토큰 길이

padded = pad_sequences( # 패딩 객체 생성
    seq, # seq (토큰)을
    maxlen=max_len, # 형식에 맞춰서
    padding="pre" # 끼어 맞춰라
)

print(padded) # 패딩 결과

# ③ 임베딩 레이어 (임베딩 객체 생성)
embedding = Embedding(
    input_dim=1000,   # 단어 사전 크기 (얼마나 많은 규칙을 정의할 수 있는가?)
    output_dim=8      # 임베딩 벡터 차원 (얼마나 규칙을 세밀하게 표현할 것인가?) 
)

embedded = embedding(padded) # 임베딩 결과
print(embedded)
