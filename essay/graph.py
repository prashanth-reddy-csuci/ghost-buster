import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from essay.data.load import generate_dataset
import dill as pickle
from utils.featurize import *

INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

best_features = open("best_features.txt").read().strip().split("\n")

with open("data/train.txt") as f:
    train_indices = list(map(int, f.read().strip().split("\n")))

data = generate_dataset(t_featurize, verbose=True)

X = normalize(np.concatenate(
    [exp_to_data[e] for e in best_features] + [data],
    axis=1
))
y = np.array([1, 0] * 2685)

X, y = X[train_indices], y[train_indices]

# Perform PCA on the dataset to extract the principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Fit a logistic regression classifier on the principal components
lr = LogisticRegression(penalty='l2', C=10, max_iter=1000)
lr.fit(X_pca, y)

# Create a mesh grid to visualize the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predictions on the mesh grid using the logistic regression classifier
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Logistic Regression Decision Boundary')
plt.savefig("pca.png")
