import os
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import tensorflow as tf
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



train_var = pd.read_csv(r"D:\COLLEGEMATERIALS\Project Me\archive\train.csv")
train_text= train_var['text'].tolist()
train_emotion = train_var['emotion'].tolist()
Train_array= np.array(train_text)

test_var = pd.read_csv(r"D:\COLLEGEMATERIALS\Project Me\archive\test.csv")
test_text= test_var['text'].tolist()
Test_array= np.array(test_text)
#Train_array= np.array(train_text)
#st=Train_array.split()
#print(train_text)
#print()



max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_text)
Train_sequences = tokenizer.texts_to_sequences(train_text)
Train_tweets = tf.keras.preprocessing.sequence.pad_sequences(Train_sequences)
print(Train_tweets)


tokenizer.fit_on_texts(test_text)
Test_sequences = tokenizer.texts_to_sequences(test_text)
Test_tweets = tf.keras.preprocessing.sequence.pad_sequences(Test_sequences )
print(Test_tweets[1])

#batch_size = 64

Train_it=tf.keras.utils.to_categorical(Train_tweets)
Test_it=tf.keras.utils.to_categorical(Test_tweets)



embed_dim = 128

lstm_out = 196

model = Sequential()


model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


Le = LabelEncoder()

y = Le.fit_transform(train_var['emotion'])


X_train, X_test, y_train, y_test = train_test_split(Train_it,y, test_size = 0.15, random_state = 42)

history=model.fit(X_train, y_train,validation_data = (X_test,y_test),epochs = 7, batch_size=32)

_,acc = model.evaluate(X_test,y_test,verbose=0)



print("Prediction: ",X_test[5:10])

print("Actual: \n",y_test[5:10])



'''
Le = LabelEncoder()

y = Le.fit_transform(train_var['emotion'])


X_train, X_test, y_train, y_test = train_test_split(Train_it,y, test_size = 0.15, random_state = 42)

embedding_layer = Embedding(1000, 64)
model1 = Sequential()
model1.add(layers.Embedding(max_words, 20,input_length = 4810)) #The embedding layer
model1.add(layers.LSTM(15,dropout=0.5)) #Our LSTM layer
model1.add(layers.Dense(3,activation='softmax'))


model1.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model1.fit(X_train, y_train, epochs=2,validation_data=(X_test, y_test),callbacks=[checkpoint1])

'''
'''
Le = LabelEncoder()

y = Le.fit_transform(train_var['emotion'])


X_train, X_test, y_train, y_test = train_test_split(Train_it,y, test_size = 0.15, random_state = 42)

#embedding_layer = Embedding(1000, 64)
model3 = Sequential()
model3.add(layers.Embedding(max_words, 40, input_length=max_len))
model3.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.MaxPooling1D(5))
model3.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.GlobalMaxPooling1D())
model3.add(layers.Dense(3,activation='softmax'))
model3.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['acc'])
history = model3.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test))
'''