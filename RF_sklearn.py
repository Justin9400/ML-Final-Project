import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Reads in the data set 
data = pd.read_csv('Top 100 Stems.csv')

columnNames = ["Stems", "Counts"]
# Extracting Attributes / Features
X = data[columnNames]

# Extracting Target / Class Labels
y = data["Class"]

# Creating Train and Test datasets, the test size is 25% and train is 75%
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 50, test_size = 0.25)

# Create Random Forest Classifier
clf = RandomForestClassifier(n_estimators = 100) 
clf.fit(X_train,y_train) 

# Predict Accuracy Score on test data set
y_pred = clf.predict(X_test)

print("Train data accuracy:", accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
print("Test data accuracy:",  accuracy_score(y_true = y_test, y_pred=y_pred))
