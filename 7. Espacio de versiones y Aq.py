from aq import AQ

# Conjunto de datos de ejemplo
data = [
    (1, 0, 0, 1, 'Positivo'),
    (0, 1, 0, 1, 'Positivo'),
    (1, 1, 0, 1, 'Negativo'),
    (0, 1, 1, 1, 'Negativo'),
    (1, 0, 1, 0, 'Negativo')
]

# Definir las etiquetas de los atributos y la etiqueta de clase
attribute_names = ['A', 'B', 'C', 'D']
class_label = 'Clase'

# Crear un objeto AQ
aq = AQ()

# Entrenar el modelo AQ con el conjunto de datos
aq.learn(data, attribute_names, class_label)

# Realizar una predicción
new_instance = (1, 1, 1, 0)
prediction = aq.predict(new_instance)
print(f'Predicción para la instancia {new_instance}: {prediction}')
