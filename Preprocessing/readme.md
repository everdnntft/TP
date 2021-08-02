### Dataset

- 모델에 넣을 학습데이터를 효율적으로 가져오기 위한 기법 중 하나
- Pipeline 구축을 통해 cpu 사용량과 gpu 사용량 스케쥴링 가능
- 학습데이터가 커서, 메모리에 올라가지 않는 상황에서 사용 가능
- Dataset API 문서 : https://www.tensorflow.org/api_docs/python/tf/data/Dataset 
- Dataset 성능향상 기법 참조 : https://www.tensorflow.org/guide/data_performance?hl=ko


### Generator

- Dataset과 마찬가지로 큰 데이터를 학습 시킬 때 사용
- 사용법은 yield Method를 사용하는 방법과 tf.keras.utils.Sequence를 사용하는 방법이 있다.
  - yield Method 사용법 : 
  - tf.keras.utils.Sequence 사용법
    - https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
