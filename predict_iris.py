import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 


def flower_prediction(llist):
    df = pd.read_csv('Iris.csv',index_col='Id')
    df.reset_index(drop=True)


    X = df.drop('Species',axis=1)
    y = df['Species']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

    model = LogisticRegression()
    model.fit(X,y)

    X_pred = np.array(llist, dtype = float)
    X_pred = X_pred.reshape((1,-1))
    predictions = model.predict(X_pred)

    return predictions

#print(flower_prediction([6.1,2.8,4.7,1.2]))



