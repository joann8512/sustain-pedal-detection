from __future__ import print_function
from __future__ import division
import os, sys
import argparse
import numpy as np
import pandas as pd
from builtins import range
from sklearn.metrics import roc_auc_score
import librosa, librosa.display
import matplotlib.pyplot as plt
#% matplotlib inline

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input, Reshape, Dropout, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # the number of the GPU
config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.7 # percentage to be used
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
set_session(tf.Session(config=config))

from kapre.time_frequency import Melspectrogram
from global_config import *



def data_gen(df_subset, ys, is_shuffle, batch_size=20):
    """Data generator.
    df_subset: pandas dataframe, with rows subset
    ys: numpy arrays, N-by-8 one-hot-encoded labels
    is_shuffle: shuffle every batch if True.
    batch_size: integer, size of batch. len(df_subset) % batch_size should be 0.
    """
    n_data = len(df_subset)
    n_batch = n_data // batch_size
    if n_data % batch_size != 0:
        print("= WARNING =")
        print("  n_data % batch_size != 0 but this code does not assume it")
        print("  so the residual {} sample(s) will be ignored.".format(n_data % batch_size))

    while True:
        for batch_i in range(n_batch):
            if is_shuffle:
                batch_idxs = np.random.choice(n_data, batch_size, replace=False)
            else:
                batch_idxs = range(batch_i * batch_size, (batch_i + 1) * batch_size)

            src_batch = np.array([np.load(os.path.join(DIR_PEDAL_ONSET_NPY,
                    df_subset.loc[df_subset.index[i]].filepath.split('.')[0]+'.npy')) for i in batch_idxs],
                    dtype=K.floatx())
            src_batch = src_batch[:, np.newaxis, :]  # make (batch, N) to (batch, 1, N) for kapre compatible

            y_batch = np.array([ys[i] for i in batch_idxs],
                               dtype=K.floatx())
            
            yield src_batch, y_batch
        
        
def get_callbacks(name,patience):
    if not os.path.exists(DIR_SAVE_MODEL):
        os.makedirs(DIR_SAVE_MODEL)    
    early_stopper = keras.callbacks.EarlyStopping(patience=patience)
    model_saver = keras.callbacks.ModelCheckpoint(os.path.join(DIR_SAVE_MODEL,"{}_best_model.h5".format(name)),
                                                  save_best_only=True)
    weight_saver = keras.callbacks.ModelCheckpoint(os.path.join(DIR_SAVE_MODEL,"{}_best_weights.h5".format(name)),
                                                   save_best_only=True,
                                                   save_weights_only=True)
    csv_logger = keras.callbacks.CSVLogger(os.path.join(DIR_SAVE_MODEL,"{}.log".format(name)))
    return [early_stopper, model_saver, weight_saver, csv_logger]


def model_multi_kernel_shape(n_out, input_shape=ONSET_INPUT_SHAPE,
                             out_activation='softmax'):
    """

    Symbolic summary:
    > c2' - p2 - c2 - p2 - c2 - p2 - c2 - p3 - d1
    where c2' -> multiple kernel shapes

    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output
    """
    audio_input = Input(shape=input_shape)

    x = Melspectrogram(n_dft=N_FFT, n_hop=HOP_LENGTH, sr=SR, n_mels=128, power_melgram=2.0, 
                       return_decibel_melgram=True)(audio_input)
    x = BatchNormalization(axis=channel_axis)(x)


    x1 = Conv2D(7, (20, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x2 = Conv2D(7, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x3 = Conv2D(7, (3, 20), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)

    x = Concatenate(axis=channel_axis)([x1, x2, x3])

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(21, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(21, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(21, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)
    x = Dropout(0.25)(x)

    x = GlobalAveragePooling2D()(x)

    out = Dense(n_out, activation=out_activation, kernel_regularizer=keras.regularizers.l2(reg_w))(x)

    model = Model(audio_input, out)

    return model


#def main():
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Training for onset classification") 
    parser.add_argument("model_name", type=str, help="model name.")  # multi_kernel
    parser.add_argument("exp", type=str, help="date")
    args = parser.parse_args()
    
    dataset_name = 'pedal-onset_npydf_small.csv'
    #model_name = args.model_name
    exp_name = 'onset_{}_{}'.format(args.model_name, args.exp)
    
    reg_w = 1e-4
    batch_size = 250
    epochs = 50
    patience = 10

    print("-" * 60)
    print("       Welcome! Lets do something deep with {}.".format(dataset_name))
    print("       I'm assuming you finished pre-processing.")
    print("       We're gonna use {} model.".format(args.model_name))
    csv_path = os.path.join(DIR_PEDAL_METADATA, dataset_name)

    tracks = pd.read_csv(csv_path)
    training = tracks.loc[tracks['category'] == 'train']
    validation = tracks.loc[tracks['category'] == 'valid']
    test = tracks.loc[tracks['category'] == 'test']

    # print("Beici: We're loading and modifying label values.")
    y_train = training.label.values
    y_valid = validation.label.values
    y_test = test.label.values

    y_train = keras.utils.to_categorical(y_train, 2)
    y_valid = keras.utils.to_categorical(y_valid, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    # callbacks
    callbacks = get_callbacks(name=exp_name, patience=patience)
    early_stopper, model_saver, weight_saver, csv_logger = callbacks

    # print("Beici: Preparing data generators for training and validation...")
    steps_per_epoch = len(y_train) // batch_size
    gen_train = data_gen(training, y_train, True, batch_size=batch_size)
    gen_valid = data_gen(validation, y_valid, False, batch_size=batch_size)
    gen_test = data_gen(test, y_test, False, batch_size=batch_size)

    # print("Beici: Getting model...")
    if args.model_name == 'multi_kernel':
        print("[x]Train with {}...".format(args.model_name))
        model = model_multi_kernel_shape(n_out=2)
    elif args.model_name == 'crnn':
        model = model_crnn_icassp2017_choi(n_out=2)
    elif args.model_name == 'cnn3x3':
        model = model_conv3x3_ismir2016_choi(n_out=2)
    elif args.model_name == 'cnn1d':
        model = model_conv1d_icassp2014_sander(n_out=2)

    # model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    print("[x]Starting to train...")
    model.fit_generator(gen_train, steps_per_epoch, epochs=epochs,
                        callbacks=callbacks,
                        validation_data=gen_valid,
                        validation_steps=len(y_valid) // batch_size)
    
    print("[x]Training is done. Loading the best weights...")
    model.load_weights(os.path.join(DIR_SAVE_MODEL,"{}_best_weights.h5".format(exp_name)))

    print("[x]Evaluating...")
    scores = model.evaluate_generator(gen_test, len(y_valid) // batch_size)
    y_pred = model.predict_generator(gen_test, len(y_valid) // batch_size)
    auc = roc_auc_score(y_valid, y_pred)

    print("Result: Done for {}!".format(model_name))
    print("        valid set loss: {}".format(scores[0]))
    print("        valid set accuracy: {}".format(scores[1]))
    print("        valid set auc: {}".format(auc))