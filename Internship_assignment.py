##############Solution code for the given task####################
'''
The approach for classification model is pretty simple:

1: Remove 3 unnecessary features like molecule name,ID,conformation name and seprate our class feature from rest of the data
2: Divide the dataset into Train and validation set with ration 80:20
3: For preprocessing, Normalization is done to make all the values lie between 0 and 1. Although normalization is not actually neceassy
    in Neural network but it's better to normalize value because it gives better result. Why? because when input value is multiplied by weights the it gives smaller value 
    which result in better gradient calculation.

4. Model architecture is simple, it has 1 input layer,2 dense layer with 2048,512 units. Why larger unit is used before and smaller later? Because i allowed the model
    to learn larger range of features at first dense layer then second dense layer having smaller units will force the model to learn only important features.
    Then Dropout layer is added as regularizer incase model starts to overfit and finally output layer with 1 unit for giving out class probability between 0-1 due to sigmoid function.
5. Loss function is Binary Crossentropy and optimizer is Adam with default learning rate (0.01). Model checkpoint is used to save best model i.e., Model with lowest validation loss.
6. For evaluation, Confusion metrix,roc score,auc score,roc curve plotting is used apart from normal accuracy metric.
7. Eval Result: Accuracy of model was around 96% while AUC score was 0.99 apart from that confusion matrix showed that model generatlization is very good.

'''



import pandas as pd
import numpy as np
from keras.layers import Dense,Input,Dropout
from keras.models import Model,load_model
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,f1_score,confusion_matrix,roc_auc_score,auc,precision_score,recall_score
import matplotlib.pyplot as plt

#Import Data
df=pd.read_csv('/home/anubhav/Downloads/musk_csv.csv')

#Drop Unnecessary columns
df=df.drop(['ID','molecule_name','conformation_name'],axis=1) 

#Some hyperparameters
EPOCHS=30
BATCH_SIZE=32

#Seprate Class feature from regressors
df_y=df['class'].values
df_X=df.drop('class',axis=1).values

#Divide data in train/val in 80:20 ratio
train_X,val_X,train_y,val_y=train_test_split(df_X,df_y,test_size=0.2)

#Normalize values
scaler=MinMaxScaler()
train_X_scaled=scaler.fit_transform(train_X)
val_X_scaled=scaler.transform(val_X)

#Model
inp=Input(shape=(166,))
x=Dense(2048,activation='relu')(inp)
x=Dense(512,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(1,activation='sigmoid')(x)
model=Model(inp,x)
model.summary()

#Model Training
ckpt_path='/home/anubhav/Documents/output/mol_classifier.h5'
mc=ModelCheckpoint(ckpt_path,monitor='val_loss',verbose=1,mode='min',
                   period=1,save_best_only=True)
rop=ReduceLROnPlateau(factor=0.2,patience=5,min_lr=0.00000001,verbose=1)

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['acc'])
history=model.fit(train_X_scaled,train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,
                  validation_data=(val_X_scaled,val_y),callbacks=[mc,rop])



##Metric Plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b',color='red', label='Training acc')
plt.plot(epochs, val_acc, 'b',color='blue', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', color='red', label='Training loss')
plt.plot(epochs, val_loss, 'b',color='blue', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#Load Best Saved Model
best_model=load_model(ckpt_path)

#Make Predictions
preds=best_model.predict(val_X_scaled)
preds=[0 if i<0.5 else 1 for i in preds]

#Eval Metrics
print('Confusion Matrix:')
print(confusion_matrix(val_y,preds))
print('ROC SCORE: {}'.format(roc_auc_score(val_y,preds)))
print('Precision Score: {:.3f}'.format(precision_score(val_y,preds)))
print('Recall Score: {:.3f}'.format(recall_score(val_y,preds)))
print('F1 Score : {:.3f}'.format(f1_score(val_y,preds)))

#Plot Roc-Curve
pred_prob=best_model.predict(val_X_scaled).ravel()
fpr,tpr,thresh=roc_curve(val_y,pred_prob)
plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='AUC Score: {:.3f}'.format(auc(fpr,tpr)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Score')
plt.legend(loc='best')
plt.show()

