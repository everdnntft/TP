import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Layer, Multiply, Dense, Flatten, LSTM, Conv1D, TimeDistributed
from tensorflow.keras.models import *
from tensorflow.keras import backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==2

        self.W_query = self.add_weight(name='att_weight_query',
                                 shape=(input_shape[0][1], input_shape[0][1]),
                                 initializer='uniform',
                                 trainable=True)
        self.W_key = self.add_weight(name='att_weight_key',
                                 shape=(input_shape[1][2], input_shape[1][1]),
                                 initializer='uniform',
                                 trainable=True)

        self.W_value = self.add_weight(name='att_weight_value',
                                   shape=(input_shape[0][1], input_shape[0][1]),
                                   initializer='uniform',
                                   trainable=True)
        
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        a = K.softmax(K.tanh(K.dot(x1, self.W_query)+ K.dot(x2, self.W_key)))
        a = K.dot(a, self.W_value)
        outputs = a * x1
        outputs = K.l2_normalize(outputs, axis=1)
        
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2]
    
# Conv_lstm
main_input = Input((15, 7, 1),name='main_input')
con1 = TimeDistributed(Conv1D(filters=10, kernel_size=3, padding='same', activation='relu', strides=1))(main_input)
con2 = TimeDistributed(Conv1D(filters=10, kernel_size=3, padding='same', activation='relu', strides=1))(con1)
#con3 = TimeDistributed(AveragePooling1D(pool_size=2))(con2)
con_out = TimeDistributed(Flatten())(con2)

lstm_out1 = LSTM(15, return_sequences=True, name='lstm_1')(con_out)
lstm_out2 = LSTM(15, return_sequences=True, name='lstm_2')(lstm_out1) # (None, 15, 15)
attention_vector = AttentionLayer()([lstm_out2,  con_out]) # (None, 15, 15), (none, 15, 70)
attention_hidden = Multiply(name='Multiply')([attention_vector, lstm_out2])
conv_lstm_out = tf.reduce_sum(attention_hidden, axis=1)


# Bilstm
auxiliary_input_w = Input((30, 1), name='auxiliary_input_w')
lstm_outw1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_w)
lstm_outw2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outw1)

auxiliary_input_d = Input((30, 1), name='auxiliary_input_d')
lstm_outd1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_d)
lstm_outd2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outd1)

x = keras.layers.concatenate([conv_lstm_out, lstm_outw2, lstm_outd2])
x = Dense(20, activation='relu')(x)
x = Dense(10, activation='relu')(x)
main_output = Dense(4, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(0.1, 0.1), name='main_output')(x)
model = Model(inputs = [main_input, auxiliary_input_w, auxiliary_input_d], outputs = main_output)
model.summary()
