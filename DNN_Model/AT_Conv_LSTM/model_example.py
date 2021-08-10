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

        self.W_0 = self.add_weight(name='att_weight0',
                                 shape=(input_shape[0][1], input_shape[0][1]),
                                 initializer='uniform',
                                 trainable=True)
        self.W_1 = self.add_weight(name='att_weight1',
                                 shape=(input_shape[1][2], input_shape[1][1]),
                                 initializer='uniform',
                                 trainable=True)

        self.W_2 = self.add_weight(name='att_weight2',
                                   shape=(input_shape[0][1], input_shape[0][1]),
                                   initializer='uniform',
                                   trainable=True)
        # self.b = self.add_weight(name='att_bias',
        #                          shape=(input_shape[0][1],),
        #                          initializer='uniform',
        #                          trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x1 = K.permute_dimensions(inputs[0], (0, 1))
        x2 = K.permute_dimensions(inputs[1][:,-1,:], (0, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x1, self.W_0)+ K.dot(x2, self.W_1)))
        a = K.dot(a, self.W_2)
        outputs = K.permute_dimensions(a * x1, (0, 1))
        # outputs = K.sum(outputs, axis=1)
        outputs = K.l2_normalize(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]

main_input = Input((15, 7, 1),name='main_input')
con1 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(main_input)
con2 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(con1)
#con3 = TimeDistributed(AveragePooling1D(pool_size=2))(con2)
con_out = TimeDistributed(Flatten())(con2)

lstm_out1 = LSTM(15, return_sequences=True, name='lstm_1')(con_out)
lstm_out2 = LSTM(15, return_sequences=True, name='lstm_2')(lstm_out1)
attention_vector = AttentionLayer()([lstm_out2[:, -1, :], con_out])
attention_mul= Multiply(name='Multiply')([attention_vector, lstm_out2])
attention_sum = tf.reduce_sum(attention_mul, axis=1)


# Bilstm
auxiliary_input_w = Input((30, 1), name='auxiliary_input_w')
lstm_outw1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_w)
lstm_outw2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outw1)

auxiliary_input_d = Input((30, 1), name='auxiliary_input_d')
lstm_outd1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_d)
lstm_outd2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outd1)

x = keras.layers.concatenate([attention_sum, lstm_outw2, lstm_outd2])
x = Dense(20, activation='relu')(x)
x = Dense(10, activation='relu')(x)
main_output = Dense(1, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(0.1, 0.1), name='main_output')(x)
model = Model(inputs = [main_input, auxiliary_input_w, auxiliary_input_d], outputs = main_output)
model.summary()
