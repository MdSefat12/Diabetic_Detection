import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

import seaborn as sns

df=pd.read_csv("diabetes.csv")
df.head()
df.tail()
df.keys()
print(df.shape)

df.corr()

df.columns
X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y=df['Outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train, y_train) 
y_prediction_nb = model_nb.predict(X_test) 
score_nb = metrics.accuracy_score(y_prediction_nb, y_test).round(4)
print("---------------------------------")
print('The accuracy of the NB is: {}'.format(score_nb))
print("---------------------------------")

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_prediction_nb)
confusion_matrix(y_test,y_prediction_nb)






sns.pairplot(df, hue="Outcome", height = 2, palette = 'colorblind');



correlation_matrix = df.corr().round(2)
plt.figure(figsize = (9, 6))
sns.heatmap(data=correlation_matrix, annot=True)