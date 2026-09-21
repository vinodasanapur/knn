import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Load the Iris dataset
iris = load_iris()

# 2. Create DataFrame for features
df = pd.DataFrame(iris.data, columns=iris.feature_names)
dfi = df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
]

# Target species labels (0: setosa, 1: versicolor, 2: virginica)
y_label = iris.target

# 3. Split data into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    dfi, y_label, test_size=0.2, random_state=42
)

# 4. Initialize and fit KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# 5. Predict test labels and calculate accuracy
pv = knn.predict(x_test)
accuracy = accuracy_score(y_test, pv)

print(f"Accuracy: {accuracy:.4f}")
