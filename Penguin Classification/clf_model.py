import pandas as pd

penguins = pd.read_csv('penguins_cleaned.csv')
df = penguins.copy()

# ordinal feature encoding
encode = ['sex', 'island']
target = 'species'

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val) -> int:
    return target_mapper[val]


df[target] = df[target].apply(target_encode)

# separating X and y
X = df.drop(target, axis=1)
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['auto','log2'],
    'criterion': ['gini', 'entropy'],
    'max_depth': [50, 100, 150],
    'min_samples_leaf': range(1, 10, 1),
    'min_samples_split': range(2,10,1)
}
grid_search = RandomizedSearchCV(clf, param_grid)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_

# train and test the model
best_clf.fit(X_train, y_train)
print(best_clf.score(X_test, y_test))

import pickle
pickle.dump(best_clf, open('penguins_clf.pkl', 'wb'))

