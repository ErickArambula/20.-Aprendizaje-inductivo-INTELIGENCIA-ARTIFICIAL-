from sklearn.datasets import load_iris
from sklearnfoil import FOILClassifier

# Cargar el conjunto de datos de Iris como ejemplo
iris = load_iris()
X, y = iris.data, iris.target

# Crear un objeto FOILClassifier
foil = FOILClassifier()

# Aprender reglas de clasificación
foil.fit(X, y)

# Obtener las reglas aprendidas
rules = foil.get_rules()

# Realizar una predicción utilizando las reglas
sample_instance = X[0]
prediction = foil.predict([sample_instance])

# Mostrar las reglas y la predicción
print("Reglas de clasificación aprendidas:")
for rule in rules:
    print(rule)
print(f"\nPredicción para la instancia: {sample_instance} - Clase: {prediction}")
