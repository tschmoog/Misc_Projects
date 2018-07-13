import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('Wholesale customers data.csv')
#Check  head data
print df.head()
#Check for missing values
print(df.isnull().values.any())
#No missing values found.

X = np.array(df.drop(['Channel', 'Region'],1))
y_chan = np.array(df.Channel)
y_region = np.array(df.Region)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y_chan, test_size= 0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
