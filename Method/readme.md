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
output = layers.Dense(1)(hidden) # output 개수: 1개 -> Units 개수

model = keras.Model(inputs=input, outputs=output)
</code>
</pre>




</br></br>
# 2. CNN (합성곱신경망, Convolution Neural Network)

* Convolution은 이미지 분석에 큰 성과를 이뤄냈다.
* 이미지는 (H, W, C) 3차원 Shape으로 C는 RGB를 나타낸다.
* 이미지를 완전연결(FC, Dense) 모델로 학습 할 경우 <br/> a. 데이터가 1차원이어야 한다. <br/> b. 고해상도 이미지 일수록 파라미터 개수가 급증한다. (H x W x C) x (output개수) <br/> c. 또한 직렬화로 인해 데이터의 공간적 형상이 무시된다.

CNN은 위와 같은 단점을 보완하면서 이미지 분석을 효율적으로 할 수 있는 방법이다.

<img src="https://user-images.githubusercontent.com/87812424/136781126-ffabc83e-1441-43e8-a305-1734df428cc6.png" width="50%" height="50%"/>

CNN은 위 그림과 같이 Kernel이라는 shared weight를 가지고 있으며, Kernel이 이동하면서 Feature map이 하나씩 채워진다.
(Kernel은 1개 Chennel에 대해서만 공유된다. filter(output chennel)개수가 10개라면 kernel은 10개 존재한다.)
정리하면 다음과 같다.
* 이미지는 인접한 공간에 대한 연관성을 가지 데이터이다.
* 이러한 공간적 연관성을 Kernel이란 Weight를 통해 stride만큼 이동하며 분석한다.
* 완전연결이 아니기에, 파라미터 개수 또한 줄일 수 있다.

아래는 (32, 32, 3) 이미지를 입력으로 받아, 개와 고양이를 분류하는 단순한 예제이다.
<pre>
<code>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(32, 32, 3)) # (32, 32) 크기의 이미지 (3은 RGB)
hidden = layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input)
fc_layer = layers.Flatten()(hidden)
fc_layer = layers.Dense(32, activation='relu')(fc_layer)
output = layers.Dense(2, activation='softmax')(fc_layer) # 분류를 위해 softmax 사용

model = keras.Model(inputs=input, outputs=output)
</code>
</pre>




</br></br>
# 3. RNN, LSTM, GRU (순환신경망)

* 자연어, 주식 등 순서가 있는 연속적인 시계열 데이터 학습에 적합한 모델이다.
* 쉽게 D<sub>t-1</sub>는 D<sub>t</sub>에 영향을 주기 때문에 이를 반영한 구조가 순환신경망이다. 

<img src="https://user-images.githubusercontent.com/87812424/137648897-85e9d5e0-bd6b-4474-9418-b721bdcc3542.jpg" width="70%" height="70%"/>


## RNN
<img src="https://user-images.githubusercontent.com/87812424/137648847-8dc6a6de-2296-4956-9dd0-d3668e0322f8.png" width="60%" height="60%"/>

* h<sub>t</sub> = tanh(W<sub>x</sub>x<sub>t</sub> + W<sub>h</sub>h<sub>t-1</sub> + bais)
* Shape 예시 및 설명</br>
  x : (i, j) , Input 데이터로 m은 timeStamp 수, n은 각 step에 들어갈 Feature 수</br>
  W<sub>x</sub> : (j, n), 학습 가능한 Weight </br>
  W<sub>h</sub> : (n, n), 학습 가능한 Weight </br>
  bais : (n, ), 편향
* 위 수식과 같이 현재의 결과(hidden == ouput)를 추출하기 위해 이전 결과를 사용한다.

<pre>
<code>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(12, 32)) # (12, 5) 과거12개, 각각 5개의 Feature를 가진 데이터 
rnn = layers.SimpleRNN(32)(input) # (32, )
output = layers.Dense(1)(rnn) # 미래 1개 예측

model = keras.Model(inputs=input, outputs=output)
</code>
</pre>


## LSTM

* 장기 의존성 문제(the problem of Long-Term Dependencies) </br>
  - 예측의 가장 중요한 정보가 시점의 앞쪽에 존재할 수 있다.
  - RNN은 충분한 기억능력을 가지지 못하기 때문에, 긴 Sequnce에서 엉뚱한 결과가 나올 수 있다.

<img src="https://user-images.githubusercontent.com/87812424/137650007-4e53043a-cb56-408b-aa63-20e99f5dd2bf.png" width="60%" height="60%"/>

* Input Gate</br>
  ![input](https://user-images.githubusercontent.com/87812424/137650844-3044e6b3-bdab-406f-9720-b1e196bf824e.png)
  - x<sub>t</sub>를 무조건적 수용이 아니라 적절히 선택하는 역할
* Output Gate</br>
  ![out](https://user-images.githubusercontent.com/87812424/137650883-80e91112-6b45-48da-82fe-05d7798c554c.png)
  - Next Timestamp에 현재의 출력이 얼마나 중요한지 조정하는 역할
* Forget Gate</br>
  ![forget](https://user-images.githubusercontent.com/87812424/137650889-632d81a4-398b-4e81-8401-8fee2c2bb3ca.png)
  - C<sub>t-1</sub>의 기억 중에서 불필요한 기억을 잊게 해주는 역할

<pre>
<code>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(12, 32)) # (12, 5) 과거12개, 각각 5개의 Feature를 가진 데이터 
rnn = layers.LSTM(32)(input) # (32, )
output = layers.Dense(1)(rnn) # 미래 1개 예측

model = keras.Model(inputs=input, outputs=output)
</code>
</pre>


## GRU

* LSTM의 장기의존성문제에 대한 해결책을 유지하면서, Hidden State를 업데이트하는 계산량을 줄이면서, 성능은 LSTM과 유사한 모델

<img src="https://user-images.githubusercontent.com/87812424/137651233-51fc4cf1-4494-4fb4-8e02-b576b9122c73.png" width="60%" height="60%"/>

* Reset Gate </br>
  <img src="https://user-images.githubusercontent.com/87812424/137651583-aa3c7406-b863-41e0-8f9b-999c0f89fa3d.png" width="20%" height="20%"/>
  - 이전 정보를 얼마나 잊어야하는지 결정하는 역할
* Update Gate </br>
  <img src="https://user-images.githubusercontent.com/87812424/137651591-cf4ab0f3-993b-4f40-b0cb-ce909436be5c.png" width="20%" height="20%"/>
  - 이전 정보를 얼마나 통과시킬지 결정하는 역할

<pre>
<code>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(12, 32)) # (12, 5) 과거12개, 각각 5개의 Feature를 가진 데이터 
rnn = layers.GRU(32)(input) # (32, )
output = layers.Dense(1)(rnn) # 미래 1개 예측

model = keras.Model(inputs=input, outputs=output)
</code>
</pre>

## LSTM(GRU) + Attention

* Query, Key, Value를 중점으로 아래 URL 참조
  - https://wikidocs.net/22893




</br></br>
# 4. GCN (그래프컨볼루션신경망, Convolution Neural Network)




