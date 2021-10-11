# 1. FC-Layer (완전연결층, Fully-Connected Layer) / Affine Layer

![dense](https://user-images.githubusercontent.com/87812424/136775670-da1505ae-dd4f-4b87-9f1d-0a657b54e220.png)

*  한층의 모든 뉴런이 다음층의 모든 뉴런과 연결된 상태
*  output의 각 element를 생성하는데 모든 input이 사용된다 하여 완전연결층이라 이름이 붙음
*  Y = f(W * X + b ) <br/> (X: input data, W: weight, b: bais, Y: output data, f: activation func)

아래는 간단한 Keras Dense 사용 예시이며, <br/>
5개의 input 데이터를 입력으로 1개의 정답을 추출하는 코드이다.<br/>
(ex. 과거 속도 5개로 미래 속도 1개 예측)<br/>
<pre>
<code>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(5, ))
hidden = layers.Dense(32, activation="relu")(input)
output = layers.Dense(1, activation="relu")(hidden)

model = keras.Model(inputs=input, outputs=output)
</code>
</pre>

# 2. CNN (합성곱신경망, Convolution Neural Network)

* 


