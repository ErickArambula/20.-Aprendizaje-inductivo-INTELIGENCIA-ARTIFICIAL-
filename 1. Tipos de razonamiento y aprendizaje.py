from sklearn.neighbors import KNeighborsClassifier

# Datos de entrenamiento (características)
X = [[1], [2], [3], [4], [5], [6]]
# Etiquetas de clase correspondientes a los datos de entrenamiento
y = [0, 0, 1, 1, 2, 2]

# Crear un clasificador K-NN con k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo
knn.fit(X, y)

# Realizar una predicción
nueva_instancia = [[2.5]]
prediccion = knn.predict(nueva_instancia)

print("Clase predicha para la nueva instancia:", prediccion)
