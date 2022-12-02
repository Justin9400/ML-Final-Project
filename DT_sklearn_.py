import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Reads in the data set 
data = pd.read_csv('Top 100 Stems.csv', encoding= "utf-8")

columnNames = []
for col in data.columns:
    if col == "Class":
        break
    else:
        columnNames.append(col)
# columnNames = ["program", "like", "peter",	"programm",	"griffin", "world",	"lacross", "hello", "track", "soccer"]
# Extracting Attributes / Features
X = data[columnNames]

# Extracting Target / Class Labels
y = data["Class"]

# Import Library for splitting data
from sklearn.model_selection import train_test_split

# Creating Train and Test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 50, test_size = 0.30)

# Creating Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

# Predict Accuracy Score
y_pred = clf.predict(X_test)
testScore1 = ["Test", "Train"]
testScore2 = [accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)), accuracy_score(y_true = y_test, y_pred=y_pred)]
print("Train data accuracy:", accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
print("Test data accuracy:", accuracy_score(y_true = y_test, y_pred=y_pred))

fig, ax = plt.subplots()
bars = ax.bar(testScore1, testScore2)
ax.bar_label(bars)
plt.title('Decision Tree Regression')
plt.show()