
# Pengambilan modul modul yang dibutuhkan 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer




# Pembacaan file csv yang digunakan untuk proses backpro
data = pd.read_csv("biji_ekstraksi.csv")



# pelabelan sebelumnya label dari data baik dan jelek di ubah ke number dari string ke 0 dan 1
le = LabelBinarizer()


# prosessing penghilangan data noise dengan membuang data tabel yang tidak diperlukan seperti Unamed: 0
# kode pre-processing data['class'] label di ubah ke number
data['class'] = le.fit_transform(data['class'])
data.drop(columns=['Unnamed: 0'], inplace=True)
data


# PEMBAGIAN DATA KEDALAM CIRI DAN KELAS MELIPUTI CIRI 'mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b'
# DAN KELAS COLUMN CLASSS
X = data[['mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b']]
Y = data['class']


# CEK PRINT DATA CLASS YAITU VARIABEL Y
Y


# PROSES PRE-PROCESSING DENGAN MEMBAGI DATA TESTING DAN TRANING
# MENGGUNAKAN LIBARRY SCKITLERN DENGAN FUNGSI TRAIN_TEST_SPLIT 
# JUMLAH DATA TRAINING 80% DARI DATA KESELURUHAN
# JUMLAH DATA TESING 20% DARI DATA KESELURUHAN 
# DISIMPAN PADA VARIBEL X_TRAIN, X_TEST UNTUK CIRI"
# DISIMPAN PADA VARIBEL Y_TRAIN, Y_TEST UNTUK OUTPUT CLASS YANG DIBUTUHKAN YAITU 0,1 ATAU BAIK/JELEK

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
X_test



# PROSESS BACKPRO MENGGUNAKAN BANTUAN LIBARAY TENSORFLOW FUNGSI MODEL SQUENTIAL
model = tf.keras.models.Sequential()
# PROSSES ARSITEKTUR BACKPRO
model.add(tf.keras.layers.Dense(units=6, activation='relu')) # INPUT LAYER  DENGAN VALUE = SEJUMLAH CIRI YAITU 6
model.add(tf.keras.layers.Dense(units=12, activation='relu')) # HIDDEN LAYER HANYA 1 PADA PROSES BACK PRO INI YANG MEMBEDAKAN DENGAN CNN DIMNA CNN BANYAK HIDDEN LAYER
model.add(tf.keras.layers.Dense(2,activation='softmax'))# OUTPUT LAYER DIGUNAKAN PROSES OUTPUT YANG DI INGINKAN DENGAN ACTIVATION FUNCTION DAN NILAI 2 SESEUAI OUTPUT YANG DINGINAN 0,1 ATAU BAIK/JELEK
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = tf.optimizers.Adam(),
              metrics=['accuracy']                  
              )
# MODEL COMPILE UNTUK PROSES TRAINING 
              


# melakukan proses training menggunakan model arsitektur backpro yang sudah dibangun dengan iterasi/perulangan epoch 100
history = model.fit(X_train,y_train,epochs=100)



# melakukan proses prekdiksi menentukan kopu jelek/baik
import numpy as np
pred = model.predict(X_test)


for i in range(len(pred)):
    # print(np.argmax(pred[i]))
    if np.argmax(pred[i]) == 0: # menggunakan nilai max dilibrary numpy jika nilai 0 baik dan 1 jelek
        print("baik")
    else:
        print('jelek')




   





