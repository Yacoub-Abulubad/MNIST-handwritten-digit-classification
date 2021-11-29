from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
 
 


def Decoding(y_pred, y_true):
  y_pred_dec = []
  y_true_dec = []
  maximum = 0
  for i in range(len(y_pred)):
    maximum = y_pred[i][0]
    for j in range(4):
      if maximum < y_pred[i][j]:
        index = j
    y_pred_dec.append(index)
    index = 0
 
  for i in range(len(y_true)):
    if y_true[i] == [1,0,0,0]:
      y_true_dec.append(0)
    
    elif y_true[i] == [0,1,0,0]:
      y_true_dec.append(1)
    
    elif y_true[i] == [0,0,1,0]:
      y_true_dec.append(2)
    
    else: y_true_dec.append(3)
  
  return y_pred_dec, y_true_dec
 
yp, yt = Decoding(y_pred, y_true)
 
 
def scoring_system(y_pred_dec, y_true_dec):
 
  confusion = confusion_matrix(y_true_dec, y_pred_dec)
 
  tp = sum(confusion[i][i] for i in range(4))
  fp = 0
  for i in range(4):
    for j in range(i+1, 4):
      fp = fp + confusion[i][j]
 
  precision = tp/(tp+fp)
 
  fn = len(y_pred)-(tp+fp)
 
  recall = tp/(tp+fn)
 
  f1_score = 2*(precision*recall)/(precision+recall)
 
  print("Precision\trecall\tf1_score")
  print(precision, recall, f1_score)
 
  df = pd.DataFrame(confusion, index = ["Spot","Healthy","Rust","White"],
                  columns = ["Spot","Healthy","Rust","White"]) 
  
  fig, ax = plt.subplots(figsize=(7,6))
  plt.xlabel("Predicted")
  ax.invert_yaxis()
  ax.xaxis.tick_top()
  ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
  ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
  plt.Axes.set_title(ax,"prediction", fontsize=15)
  plt.ylabel("Actual", fontsize=15)
  plt.ylim(0,4)
  plt.xlim(4,0)
  sn.heatmap(df, annot=True,fmt="d", cmap='Greens', linecolor='black', linewidths=1)
  plt.ylabel('Actual', rotation=0, va='center')
  plt.yticks(rotation=0)
  
scoring_system(yp, yt)
 
def FC_graphing(history):  
  plt.plot(history.history['FC_precision'])
  plt.plot(history.history['val_FC_precision'])
  plt.title('model precision')
  plt.ylabel('Precision')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

  plt.plot(history.history['FC_recall'])
  plt.plot(history.history['val_FC_recall'])
  plt.title('model recall')
  plt.ylabel('Recall')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
 
def AE_graphing(history):  
  plt.plot(history.history['Dec_mean_absolute_error'])
  plt.plot(history.history['val_Dec_mean_absolute_error'])
  plt.title('Model MAE')
  plt.ylabel('Mean Absolute Error')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
 

