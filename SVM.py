import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('voice.csv')

label = data.iloc[:,-1:].values

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],label,test_size=0.33,random_state=0)

svc = SVC(kernel = 'poly')

svc.fit(x_train,y_train)

result = svc.predict(x_test)

cm = confusion_matrix(y_test,result)
print(cm)

accuracy = accuracy_score(y_test, result)
print(accuracy)
