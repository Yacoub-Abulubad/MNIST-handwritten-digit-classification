#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.metrics import confusion_matrix

#%%
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# %%
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])

# %%
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
# %%
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1] * x_train.shape[2]
# %%
y_test = keras.utils.to_categorical(y_test, 10)
y_train = keras.utils.to_categorical(y_train, 10)
#%%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
# %%
model = keras.Sequential()
optimizer = keras.optimizers.Adadelta(learning_rate=0.001)
loss = keras.losses.categorical_crossentropy
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
# %%


def scoring_system(y_pred, y_true):
    
    y_pred_dec = []
    y_true_dec = []
    maximum = 0
    index = 0
    for i in range(len(y_pred)):
      maximum = y_pred[i][0]
      for j in range(10):
        if maximum < y_pred[i][j]:
          index = j
      y_pred_dec.append(index)
      index = 0
    
    index = 0
    for i in range(len(y_true)):
      maximum = y_true[i][0]
      for j in range(10):
        if maximum < y_true[i][j]:
          index = j
      y_true_dec.append(index)
      index = 0

    confusion = confusion_matrix(y_true_dec, y_pred_dec)

    tp = sum(confusion[i][i] for i in range(10))
    fp = 0
    for i in range(10):
      for j in range(i+1, 10):
        fp = fp + confusion[i][j]

    precision = tp/(tp+fp)

    fn = len(y_pred_dec)-(tp+fp)

    recall = tp/(tp+fn)

    f1_score = 2*(precision*recall)/(precision+recall)

    print(precision, recall, f1_score)

    df = pd.DataFrame(confusion, index = "0123456789",
                    columns = "0123456789") 
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    sn.heatmap(df, annot=True, cmap='Greens')

    return precision, recall, f1_score
#%%
model.compile(loss=loss,optimizer=optimizer, metrics=scoring_system)
# %%
model_log = model.fit(x_train,y_train, epochs=10,validation_data=(x_val,y_val))

# %%
