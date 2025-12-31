# RNN : 앞에서 본 것을 머릿속에 저장하는 인공 신경망 (다음 단어와 문장을 예측하기 좋음) 
# ANN : 앞에서 본 것을 기억하지 못하고 그대로 문제를 계산하는 인공 신경망 (한방을 노리기 좋음)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np

# ① 토큰화
tokenizer = Tokenizer(
    num_words=1000,
    oov_token="<OOV>"
)

texts = [
    "Hello World",                # 1 (긍정적)
    "I'm very handsome",          # 1 (긍정적)
    "I'm not like you",           # 0 (부정적)
    "I hate you",                 # 0 (부정적)
    "you won't succeed",          # 0 (부정적)
    "I love you",                 # 1 (긍정적)
    "you will definitely succeed",# 1 (긍정적)
    "I'm happy",                  # 1 (긍정적)
    "I'm sad",                    # 0 (부정적)
    "This is terrible",           # 0 (부정적)
    "This is amazing",            # 1 (긍정적)
]

y = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1])

tokenizer.fit_on_texts(texts)

seq = tokenizer.texts_to_sequences(texts)
print("토큰화 결과:", seq)

# ② 패딩
max_len = 5

padded = pad_sequences(
    seq,
    maxlen=max_len,
    padding="pre"
)

# ③ RNN 모델 정의 (LSTM)
model = Sequential([
    Embedding(
        input_dim=1000,
        output_dim=8,
        input_length=max_len
    ),
    LSTM(16),  # LSTM 레이어로 시퀀스의 의미를 기억
    Dense(1, activation="sigmoid")  # 이진 분류 (긍정/부정)
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

model.summary()

# ④ 모델 가동
x = padded

model.fit(
    x,
    y,
    epochs=70,
    verbose=1
)

# ⑤ 특정 문장의 예측 확률 확인
new_text = ["I love you"]  # 예측할 문장

# ① 새로운 문장 토큰화
new_seq = tokenizer.texts_to_sequences(new_text)
print("새로운 토큰화 결과:", new_seq)

# ② 새로운 문장에 패딩 적용
new_padded = pad_sequences(
    new_seq,
    maxlen=max_len,
    padding="pre"
)
print("새로운 패딩 결과:", new_padded)

# ③ 예측 (긍정적일 확률)
new_pred = model.predict(new_padded)

# 예측 확률 출력
print(f"I love you의 긍정 확률 : {new_pred}")
