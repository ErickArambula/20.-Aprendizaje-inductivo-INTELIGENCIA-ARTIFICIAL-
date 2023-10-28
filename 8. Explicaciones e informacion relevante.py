from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.tree import export_text

# Cargar el conjunto de datos de Iris como ejemplo
iris = load_iris()
X, y = iris.data, iris.target

# Entrenar un modelo de árbol de decisión
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Obtener una explicación para una instancia específica
instance = X[0]  # Por ejemplo, tomamos la primera instancia del conjunto de datos
explanation = export_text(clf, feature_names=iris.feature_names)
print(f"Explicación para la instancia: {instance}\n")
print(explanation)
