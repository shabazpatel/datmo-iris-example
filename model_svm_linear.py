import os
import pickle
import numpy as np

import datmo

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_helper import plot_decision_regions

# Using Support Vector Machine model
from sklearn.svm import SVC

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


# Splitting data into 70% training and 30% test data:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# Standardizing the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Training a SVM classifeir with linear kernel
config = {'algorithm': 'svm', 'kernel': 'linear'}
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

y_train_pred = svm.predict(X_train_std)
y_test_pred = svm.predict(X_test_std)

stats = {"Train Accuracy": '%.2f' % accuracy_score(y_train, y_train_pred), "Test Accuracy": '%.2f' % accuracy_score(y_test, y_test_pred)}
print(stats)

# saving model file
model_filename = os.path.join('model.pkl')
pickle.dump(svm, open(model_filename, 'wb'))

# saving plot
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plot.png', dpi=300)

datmo.snapshot.create(message="svm with linear", config=config, stats=stats)