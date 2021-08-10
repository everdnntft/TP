## AT_Conv_LSTM </br>: A Hybrid Deep Learning Model With Attention-Based Conv-LSTM Networks for Short-Term Traffic Flow Prediction

</br>


### 1. 자료
 - 논문 URL
   - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9112272&casa_token=vQ8j3mkO3AoAAAAA:2JJEJL37JmL9c33c0Ke1VxwqQoTmDjB-7DdMJycibMVV3-PzpSh1yTpqg97K7K8UbZd7UpwijaiQ
 - Code URL
   - https://github.com/suprobe/AT-Conv-LSTM

</br>

### 2. 개요
 - Traffic Frocasting을 위하여, [속도, 일별 패턴, 주별 패턴] 데이터를 사용하여, 미래 Traffic을 예측합니다.
   - 속도 데이터(15, sensor개수, 1) : 링크별, 5분 단위 과거 75분(15개) 속도 데이터.
   - 일별 패턴(30, 1) : sensor별, 현재 시간 기준 과거 75분(15개)/미래 75분(15개) 일별 패턴 속도 데이터 
   - 주별 패턴(30, 1) : sensor별, 현재 시간 기준 과거 75분(15개)/미래 75분(15개) 주별 패턴 속도 데이터 </br> 
   - 예측 데이터 : 미래 5분, 15분, 30분, 60분

</br>

### 3. 모델 구조
![image](https://user-images.githubusercontent.com/87812424/128858004-b4dbdac8-aed0-4481-97df-cb506acfd0cc.png)
![image](https://user-images.githubusercontent.com/87812424/128858098-3a8111c3-3df1-4ac9-b855-7df8b9b45259.png)
 - [Fig1]
   - Conv_LSTM Input : 현재 속도 데이터
   - Bi-LSTM Input : 일별 패턴
   - bi-LSTM input : 주별 패턴
 - [Fig2]
   - 


