import numpy as np
import pandas as pd
from keras.models import *
from keras_self_attention import SeqSelfAttention
from calculate_performance import performance_fun


file_p  = '../data/test/3mer_positive_test_datavec.csv'
file_n = '../data/test/3mer_negative_test_datavec.csv'


p_test = pd.read_csv(file_p, header=None, index_col=[0])
n_test = pd.read_csv(file_n, header=None, index_col=[0])


X_test = np.vstack((p_test, n_test))
print(X_test.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1).astype('float32')
Y_test = np.array([1] * p_test.shape[0] + [0] * n_test.shape[0], dtype='float32')

self_attention = SeqSelfAttention(attention_activation='softmax')
model = load_model('../model/3fold_model.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
predictions = model.predict(X_test)

array = pd.DataFrame(predictions)
print(array)
code_file = 'independent_test_data_predicted_score.csv'
array.to_csv(code_file, index=False, mode='a', header=False)



TP, TN, FP, FN, Sn, Sp, Acc, Mcc, precision, f1_score, auc = performance_fun(predictions, Y_test)
print('test:', 'TP', 'TN', 'FP', 'FN', 'Sn', 'Sp', 'Acc', 'Mcc', 'precision', 'f1_score', 'auc')
print(TP, TN, FP, FN, Sn, Sp, Acc, Mcc, precision, f1_score, auc)



###MSE
# print(metrics.mean_squared_error(Y_test, predictions))





