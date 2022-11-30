import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Reads in the data set 
data = pd.read_csv('Top 100 Stems.csv')

# columnNames = ["claim", "prize", "tone", "guarante", "cs", "Ã¥Â£1000", "rington", "150ppm", ">", "150p", "entri", "mob", "16+", "weekli", "18", "poli", "500", "valid", 
# "Ã¥Â£100", "bonu", "8007", "sae", "Ã¥Â£5000", "Ã¥Â£2000", "Ã¥Â£1.50", "Ã¥Â£500", "86688", "unsubscrib", "18+", "http", "750", "12hr", "pobox", "winner", "mobileupd8", 
# "identifi", "8000930705", "150p/msg", "camcord", "8000839402", "Ã¥Â£250", "freemsg", "10p", "wkli", "savamob", "quiz", "87066", "ltd", "Ã¥Â£350", "800", "gt", "lt", 
# "lor", "da", "say", "later", "Ã¬_", "said", "amp", "sleep", "morn", "sure", "lol", "anyth", "smile", "watch", "someth", "\\", "finish", "went", "gon", "plan", "alway", 
# "nice", "gud", "dun", "told", "Ã¬Ã¯", "mean", "haha", "happen", "thk", "fine", "fuck", "lunch", "eat", "car", "job", "bit", "worri", "yup", "long", "dat", "problem", 
# "that", "quit", "wonder", "lar", "liao", "kiss"]
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
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 50, test_size = 0.25)

# Creating Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

# Predict Accuracy Score
y_pred = clf.predict(X_test)
print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))