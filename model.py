import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import (
    l2, 
    l1, 
    l1_l2
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import (
    activations, 
    initializers, 
    regularizers, 
    constraints
)
from attention_layer import Attention, attention_flatten
def build_model(training_pep, training_mhc):
    filters, kernel = 256, 5
    input_pep = Input(shape=(np.shape(training_pep[0])), name = 'peptide')

    conv_pep = Conv1D(filters = filters, kernel_size = kernel, activation = 'relu', 
                        padding = 'same',name = 'conv1_pep')

    conv_pool_out_pep = MaxPooling1D()
    conv_pool_dropout1_pep = Dropout(0.6)
    conv_pool_dropout2_pep = Dropout(0.5)

    decoder_pep = Attention(hidden=256, activation = 'linear')
    dense1_pep = Dense(1)
    
    output1_pep = conv_pep(input_pep)
    output2_pep = conv_pool_out_pep(output1_pep)
    output3_pep = conv_pool_dropout1_pep(output2_pep)

    att_decoder_pep = decoder_pep(output3_pep)
    output4_pep = attention_flatten(output3_pep.shape[2])(att_decoder_pep)
    
    output5_pep = dense1_pep(conv_pool_dropout2_pep(Flatten()(output3_pep)))
    output_all_pep = concatenate([output4_pep, output5_pep])
    output6_pep = Dense(1)(output_all_pep)

    #output_final_pep = Activation('sigmoid')(output6_pep)

    input_mhc = Input(shape=(np.shape(training_mhc[0])), name = 'mhc')

    conv_mhc = Conv1D(filters = filters, kernel_size = kernel, activation = 'relu', 
                        padding = 'same',kernel_constraint = MaxNorm(3), 
                        name = 'conv1_mhc')

    conv_pool_out_mhc = MaxPooling1D()
    conv_pool_dropout1_mhc = Dropout(0.6)
    conv_pool_dropout2_mhc = Dropout(0.5)

    decoder_mhc = Attention(hidden=256, activation = 'linear')
    dense1_mhc = Dense(1)
    
    output1_mhc = conv_mhc(input_mhc)
    output2_mhc = conv_pool_out_mhc(output1_mhc)
    output3_mhc = conv_pool_dropout1_mhc(output2_mhc)

    att_decoder_mhc = decoder_mhc(output3_mhc)
    output4_mhc = attention_flatten(output3_mhc.shape[2])(att_decoder_mhc)
    
    output5_mhc = dense1_mhc(conv_pool_dropout2_mhc(Flatten()(output3_mhc)))
    output_all_mhc = concatenate([output4_mhc, output5_mhc])
    output6_mhc = Dense(1)(output_all_mhc)

    #output_final_mhc = Activation('sigmoid')(output6_mhc)

    combinedOutput = concatenate([output6_pep, output6_mhc])
    combinedDenseOutput = Dense(1)(combinedOutput)
    finalCombinedOutput = Activation('sigmoid')(combinedDenseOutput)

    model = Model(inputs = ([input_pep, input_mhc]), outputs = finalCombinedOutput)
    opt = Adam(learning_rate = 1e-3)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy', 'AUC'])
    #model.summary()
    return model