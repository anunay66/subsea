
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(color_codes=True)
import pickle
data = pd.read_csv(r'C:\Users\Anunay\Desktop\TIP\db1csv 2.csv')
data = data.dropna()
data.isnull().sum()
data1 = data.drop(columns=['Name','Owner', 'URL', 'Suppliers', 'Landing Points', 'Continent'], axis=1)
data1.replace( {"Cable Length": {"n.a.": 0}}, inplace = True )

data1 = data1.astype(str) 

#converting dataframe into string datatype to solve a value error

from sklearn.model_selection import train_test_split




from sklearn.model_selection import train_test_split
Y = data1['Status']
X = data1.drop(columns=['Status'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=9)


## 4. Decision tree classification

from sklearn.tree import DecisionTreeClassifier

# We define the model
dtcla = DecisionTreeClassifier(random_state=9)

# We train model
dtcla.fit(X_train, Y_train)

# We predict target values
Y_predict4 = dtcla.predict(X_test)
# Test score
score_dtcla = dtcla.score(X_test, Y_test)
print(score_dtcla)
# Make pickle file of our model
pickle.dump(dtcla, open("modelL.pkl", "wb"))