# 1. FC-Layer (완전연결층, Fully-Connected Layer)

![dense](https://user-images.githubusercontent.com/87812424/136775670-da1505ae-dd4f-4b87-9f1d-0a657b54e220.png)

*  한층의 모든 뉴런이 다음층의 모든 뉴런과 연결된 상태
*  output의 각 element를 생성하는데 모든 input이 사용된다 하여 완전연결층이라 이름이 붙음
*  Y = f(W * X + b ) <br/> (X: input data, W: weight, b: bais, Y: output data, f: activation func)
*  파라미터 개수: input개수 x output개수 + output개수(bais)

아래는 간단한 Keras Dense 사용 예시이며, <br/>
5개의 input 데이터를 입력으로 1개의 정답을 추출하는 코드이다.<br/>
(ex. 과거 속도 5개로 미래 속도 1개 예측)<br/>
<pre>
<code>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(5, )) # input 개수: 5개
hidden = layers.Dense(32, activation="relu")(input) # output 개수: 32개 -> Units 개수
output = layers.Dense(1, activation="relu")(hidden) # output 개수: 1개 -> Units 개수

model = keras.Model(inputs=input, outputs=output)
</code>
</pre>



# 2. CNN (합성곱신경망, Convolution Neural Network)

* Convolution은 이미지 분석에 큰 성과를 이뤄냈다.
* 이미지는 (H, W, C) 3차원 Shape으로 C는 RGB를 나타낸다.
* 이미지를 완전연결(FC, Dense) 모델로 학습 할 경우 <br/> a. 데이터가 1차원이어야 한다. <br/> b. 고해상도 이미지 일수록 파라미터 개수가 급증한다. (H x W x C) x (output개수) <br/> c. 또한 직렬화로 인해 데이터의 공간적 형상이 무시된다.

CNN은 위와 같은 단점을 보완하면서 이미지 분석을 효율적으로 할 수 있는 방법이다.

<img src="https://user-images.githubusercontent.com/87812424/136781126-ffabc83e-1441-43e8-a305-1734df428cc6.png" width="50%" height="50%"/>

CNN은 위 그림과 같이 Kernel이란 shared weight를 가지고 있으며, Kernel이 이동하면서 Feature map이 하나씩 채워진다.
(Kernel은 1개 Chennel에 대해서만 공유된다. filter(output chennel)개수가 10개라면 kernel은 10개 존재한다.)
정리하면 다음과 같다.
* 이미지는 인접한 공간에 대한 연관성을 가지 데이터이다.
* 이러한 공간적 연관성을 Kernel이란 Weight를 통해 stride만큼 이동하며 분석한다.
* 완전연결이 아니기에, 파라미터 개수 또한 줄일 수 있다.
