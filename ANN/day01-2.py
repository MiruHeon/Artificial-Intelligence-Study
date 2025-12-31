# pip install tensorflow [텐서플로 토큰화 라이브러리]
from tensorflow.keras.preprocessing.text import Tokenizer

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
