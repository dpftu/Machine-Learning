
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier)


df = pd.read_csv('wine.csv', header = None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
le = LabelEncoder()
le.fit_transform(y)
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size = 0.3,
    random_state = 5,
    stratify = y

)

tree = DecisionTreeClassifier(

    criterion = 'gini',
    max_depth = 1,
    random_state = 5

)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)

bag = BaggingClassifier(

    base_estimator = tree,
    n_estimators = 500,
    bootstrap = True,
    bootstrap_features = False,
    n_jobs = 1,
    random_state = 1

)
bag.fit(X_train, y_train)

ada = AdaBoostClassifier(

    base_estimator = tree,
    n_estimators = 500,
    learning_rate = 0.01,
    random_state = 5

)

ada.fit(X_train, y_train)
ada.score(X_train, y_train)


print(f'adaptive boosting: {ada.score(X_train, y_train):.{4}}')
print(f'adaptive boosting: {ada.score(X_test, y_test):.{4}}')
print(f'bagging train: {bag.score(X_train, y_train):.{4}}')
print(f'bagging train: {bag.score(X_test, y_test):.{4}}')
print(f'unpruned tree train: {tree.score(X_train, y_train):.{3}}')
print(f'unpruned tree test: {tree.score(X_test, y_test):.{3}}')








