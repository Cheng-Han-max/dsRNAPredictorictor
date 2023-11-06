import keras
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from datetime import datetime
from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from calculate_performance import performance_fun

seed = 7
np.random.seed(seed)




def model_framework(shape):
    input = Input(shape=shape)
    conv = Convolution1D(filters=64, kernel_size=3, padding='valid', activation='relu')(input)
    pool = MaxPooling1D(pool_size=5)(conv)
    bn1 = BatchNormalization()(pool)
    dt = Dropout(0.2)(bn1)
    bi_lstm = Bidirectional(LSTM(50, return_sequences=True))(dt)
    bn2 = BatchNormalization()(bi_lstm)
    dt = Dropout(0.5)(bn2)
    self_attention = SeqSelfAttention(attention_activation='softmax')(dt)
    fl = Flatten()(self_attention)
    dense = Dense(16, activation='sigmoid')(fl)
    output = Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=input, outputs=output)

    print(model.summary())
    return model

t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print('Start time：' + t1)

code_file_p1 = '../training_vec/3mer_positive_train_datavec.csv'
code_file_n1 = '../training_vec/3mer_negative_train_datavec.csv'


p_3mer = pd.read_csv(code_file_p1, header=None, index_col=[0])
n_3mer = pd.read_csv(code_file_n1, header=None, index_col=[0])

X = np.vstack((p_3mer, n_3mer))
print(X.shape)
Y = np.array([1] * p_3mer.shape[0] + [0] * n_3mer.shape[0], dtype='float32')

skf = StratifiedKFold(n_splits=5, shuffle=True)


train_TP = []
train_TN = []
train_FP = []
train_FN = []
train_Sn = []
train_Sp = []
train_Acc = []
train_Mcc = []
train_precision = []
train_f1_score = []
train_auc = []

val_TP = []
val_TN = []
val_FP = []
val_FN = []
val_Sn = []
val_Sp = []
val_Acc = []
val_Mcc = []
val_precision = []
val_f1_score = []
val_auc = []

k = 1
for train_index, val_index in skf.split(X, Y):
    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1).astype('float32')

    train_shape = (x_train.shape[1], 1)
    cnn = model_framework(train_shape)
    cnn.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['binary_accuracy'])
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1)
    earlyStop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    callbacks = [earlyStop, lr_reduce]

    cnn.fit(x_train, y_train, validation_data=[x_val, y_val], callbacks=callbacks, epochs=320, batch_size=32, verbose=2, shuffle=True)
    model_file = 'model/' + str(k) + 'fold_model.h5'
    cnn.save(model_file)



    y_train_pred = cnn.predict(x_train)
    TP1, TN1, FP1, FN1, Sn1, Sp1, Acc1, Mcc1, precision1, f1_score1, auc1 = performance_fun(y_train_pred, y_train)
    print('train:', 'TP', 'TN', 'FP', 'FN', 'Sn2', 'Sp', 'Acc', 'Mcc', 'precision', 'f1_score', 'auc')
    print(TP1, TN1, FP1, FN1, Sn1, Sp1, Acc1, Mcc1, precision1, f1_score1, auc1)
    train_TP.append(TP1)
    train_TN.append(TN1)
    train_FP.append(FP1)
    train_FN.append(FN1)
    train_Sn.append(Sn1)
    train_Sp.append(Sp1)
    train_Acc.append(Acc1)
    train_Mcc.append(Mcc1)
    train_precision.append(precision1)
    train_f1_score.append(f1_score1)
    train_auc.append(auc1)


    y_val_pred = cnn.predict(x_val)
    TP2, TN2, FP2, FN2, Sn2, Sp2, Acc2, Mcc2, precision2, f1_score2, auc2 = performance_fun(y_val_pred,  y_val)
    print('val:', 'TP', 'TN', 'FP', 'FN', 'Sn2', 'Sp', 'Acc', 'Mcc', 'precision', 'f1_score', 'auc')
    print(TP2, TN2, FP2, FN2, Sn2, Sp2, Acc2, Mcc2, precision2, f1_score2, auc2)
    val_TP.append(TP2)
    val_TN.append(TN2)
    val_FP.append(FP2)
    val_FN.append(FN2)
    val_Sn.append(Sn2)
    val_Sp.append(Sp2)
    val_Acc.append(Acc2)
    val_Mcc.append(Mcc2)
    val_precision.append(precision2)
    val_f1_score.append(f1_score2)
    val_auc.append(auc2)
    k = k + 1



print('train_mean:')
print('train_meanTP', 'train_meanTN', 'train_meanFP', 'train_meanFN',
      'train_meanSN', 'train_meanSP', 'train_meanACC', 'train_meanMCC',
      'train_meanPrecision', 'train_meanF1_score', 'train_meanauc')
print(np.mean(train_TP), np.mean(train_TN), np.mean(train_FP), np.mean(train_FN),
      format(np.mean(train_Sn), '.4f'), format(np.mean(train_Sp), '.4f'), format(np.mean(train_Acc), '.4f'), format(np.mean(train_Mcc), '.4f'),
      format(np.mean(train_precision), '.4f'), format(np.mean(train_f1_score), '.4f'), format(np.mean(train_auc), '.4f'))

print('val_mean:')
print('val_meanTP', 'val_meanTN', 'val_meanFP', 'val_meanFN',
      'val_meanSN', 'val_meanSP', 'val_meanACC', 'val_meanMCC',
      'val_meanPrecision', 'val_meanF1_score', 'val_meanauc')
print(np.mean(val_TP), np.mean(val_TN), np.mean(val_FP), np.mean(val_FN),
      format(np.mean(val_Sn), '.4f'), format(np.mean(val_Sp), '.4f'), format(np.mean(val_Acc), '.4f'), format(np.mean(val_Mcc), '.4f'),
      format(np.mean(val_precision), '.4f'), format(np.mean(val_f1_score), '.4f'), format(np.mean(val_auc), '.4f'))



t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print('end time：' + t2)
