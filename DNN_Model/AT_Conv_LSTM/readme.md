## AT_Conv_LSTM </br>: A Hybrid Deep Learning Model With Attention-Based Conv-LSTM Networks for Short-Term Traffic Flow Prediction

</br>


### 1. 자료
 - 논문 URL
   - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9112272&casa_token=vQ8j3mkO3AoAAAAA:2JJEJL37JmL9c33c0Ke1VxwqQoTmDjB-7DdMJycibMVV3-PzpSh1yTpqg97K7K8UbZd7UpwijaiQ
 - Example Code
   - model_example.py 참조
   - 논문에 설명된 구조로 예시 모델 작성  
</br>

### 2. 개요
 - Traffic Frocasting을 위하여, [속도, 일별 패턴, 주별 패턴] 데이터를 사용하여, 미래 Traffic을 예측합니다.
   - 속도 데이터(15, sensor개수, 1) : sensor별, 5분 단위 과거 75분(15개) 속도 데이터.
   - 일별 패턴(30, 1) : sensor별, 현재 시간 기준 과거 75분(15개)/미래 75분(15개) 일별 패턴 속도 데이터 
   - 주별 패턴(30, 1) : sensor별, 현재 시간 기준 과거 75분(15개)/미래 75분(15개) 주별 패턴 속도 데이터 </br> 
   - 예측 데이터 : 미래 5분, 15분, 30분, 60분

</br>

### 3. 모델 구조 및 설명
![image](https://user-images.githubusercontent.com/87812424/128858004-b4dbdac8-aed0-4481-97df-cb506acfd0cc.png)
![image](https://user-images.githubusercontent.com/87812424/128858098-3a8111c3-3df1-4ac9-b855-7df8b9b45259.png)
 - [Fig1]
   - Input
     - Conv_LSTM Input : 현재 속도 데이터
     - Bi-LSTM Input : 일별 패턴
     - bi-LSTM input : 주별 패턴
   - 각 module의 output을 concat하고, FF Layer 걸쳐 최종 예측 데이터 생성
 - [Fig2] </br> 
   ![image](https://user-images.githubusercontent.com/87812424/128891404-1eb2113d-8d1b-4d93-a9d3-ee571b416568.png) </br> 
   ![image](https://user-images.githubusercontent.com/87812424/128891439-ce3e257c-baab-4c49-9867-af6d086f8529.png) </br> 
   ![image](https://user-images.githubusercontent.com/87812424/128891468-4c43d28e-991f-477a-b9fb-de001075a1df.png) </br> 

   - attention vector
     - Convolution Layer의 output -> shape (None, 15, 105)
     - LSTM의 output -> shape (None, 15, 15)
     - 위 두 tensor를 사용하여, 각 Time Step(axis=1) 별로 attention vector 생성 후 normalize -> shape (None, 15, 15) 
   - 각 step 별 attention vector를 LSTM의 각 step 별 output에 element wise하게 multiply -> shape (None, 15, 15)
   - 각 step의 위 결과를 모두 add하여 Conv_lstm의 ouput 생성 -> shape (None, 15)


