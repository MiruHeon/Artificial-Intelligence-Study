# RNN : 앞에서 본 것을 머릿속에 저장하는 인공 신경망 (다음 단어와 문장을 예측하기 좋음) 
# ANN : 앞에서 본 것을 기억하지 못하고 그대로 문제를 계산하는 인공 신경망 (한방을 노리기 좋음)

# 토큰화
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 레이어
from tensorflow.keras.layers import Embedding, LSTM, Dense # LSTM : 단어에서 얻은 기억을 다음 단어로 전달해주는 알고리즘 
from tensorflow.keras.models import Sequential

import numpy as np

# ____________________________________________________

# ① 토큰화
tokenizer = Tokenizer(
    num_words=1000,
    oov_token="<OOV>"
)

texts = [
    "Hello World",
    "Hello AI",
    "AI is future",
    "I love AI",
    "World is big",
    "I'm very handsome",
    "I like you"
]

tokenizer.fit_on_texts(texts)

seq = tokenizer.texts_to_sequences(["Hello World AI."])
print("토큰:", seq)

# ② 패딩
max_len = 5

padded = pad_sequences(
    seq,
    maxlen=max_len,
    padding="pre"
)

print("패딩:", padded)

# ③ RNN 모델 정의 (LSTM)
model = Sequential([
    Embedding(
        input_dim=1000,
        output_dim=8,
        input_length=max_len
    ),
    LSTM(16), # 지금까지 본 모든 단어와 시점의 기억을 16차원의 벡터로 압축해서 정리해라
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

model.summary()

# ④ 모델 가동
x = padded
y = np.array([1])

model.fit(
    x,
    y,
    epochs=10,
    verbose=1
)

# ⑤ 예측
pred = model.predict(x)
print("예측값:", pred)
