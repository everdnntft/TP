#### 아래 기법 모두 데이터 Load와 전처리를 모두 같이 진행 할 수 있지만, <br/>모델이 학습되는 과정에서 처리되는 작업이기 때문에 복잡한 전처리는 더 앞단에서 미리 처리하는 것을 추천합니다.

## Dataset

- 모델에 넣을 학습데이터를 효율적으로 가져오기 위한 기법 중 하나
- Pipeline 구축을 통해 cpu 사용량과 gpu 사용량 스케쥴링 가능
- 학습데이터가 커서, 메모리에 올라가지 않는 상황에서 사용 가능
- Dataset API 문서
  - https://www.tensorflow.org/api_docs/python/tf/data/Dataset 
- Dataset 성능향상 기법 참조
  - https://www.tensorflow.org/guide/data_performance?hl=ko


## Generator

- Dataset과 마찬가지로 큰 데이터를 학습 시킬 때 사용
- 사용법은 yield Method를 사용하는 방법과 tf.keras.utils.Sequence를 사용하는 방법이 있다.
  - yield Method 사용법 예시
    - <pre>
      <code>
      def generate_arrays_from_file(path):
            while 1:
                f = open(path)
                for line in f:
                    # create numpy arrays of input data
                    # and labels, from each line in the file
                    x1, x2, y = process_line(line)
                    yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                f.close()

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
        </code>
        </pre>
  - tf.keras.utils.Sequence 사용법
    - https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    - http://www.kwangsiklee.com/tag/model-fit_generator/
    - 코드 부연 설명
      <pre>
      <code>
      class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, ....):
          ....
          # file_path, file_idx, batch_size, Shuffle 여부 등의 학습데이터 생성에 필요한 정보 
          
        def __len__(self):
          ....
          # Generator는 total train/valid/test size를 알려주지 않으면, 스스로 알지 못합니다.
          # Total Size를 모르면 몇 step(total_size / batchsize)을 돌아야 1 epoch이 되는지 알지 못합니다.
          # 따라서 개발자는 미리 데이터의 total_size를 알아야하고, 해당 method의 output으로 step 수를 return 합니다.
            
        def __getitem__(self, index):
          ....
          # 학습데이터와 정답데이터를 batch_size만큼 생성해 return 합니다.
          # 해당 데이터로 모델이 학습됩니다.
        
        def on_epoch_end(self):
          ....
          # __len__() Method에서 정의한 step 수 만큼 돌면 1 epoch가 끝난 것이고, 1 epoch가 끝났을 때, 해당 함수가 호출됩니다.
          # 해당 함수는 return값은 없으며, 보통 file_idx를 shuffle하여, 다음 epoch에서의 학습 순서를 변경해줍니다.
          
      </code>
      </pre>
