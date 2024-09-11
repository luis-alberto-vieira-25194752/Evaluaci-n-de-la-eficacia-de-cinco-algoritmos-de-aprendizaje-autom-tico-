# Evaluación de la eficacia de cinco algoritmos de aprendizaje automático en la predicción del cáncer de mama

## Introducción 
Este proyecto tiene como objetivo utilizar cinco algoritmos de aprendizaje automático (regresión logística, árboles de decisión, bosques aleatorios, máquinas de vectores de soporte y naive bayes) para la predicción y el diagnóstico del cáncer de mama, utilizando los datos de la Universidad de Wisconsin. A través de un análisis exhaustivo de los datos y la aplicación de técnicas avanzadas, buscamos identificar el modelo más efectivo para esta tarea crítica.

## Descripción del proyecto 
El proyecto se centra en la detección de cáncer de mama mediante el uso de cinco algoritmos de aprendizaje automático. Utilizando el conjunto de datos de la Universidad de Wisconsin, se realiza una exploración profunda de los datos, incluyendo la búsqueda de valores perdidos y duplicados, y la visualización de correlaciones entre las variables predictoras mediante gráficos de pares y mapas de calor. Basándose en estas correlaciones, se seleccionaron las variables más relevantes para construir los modelos de aprendizaje automático. Posteriormente, los datos se dividen en conjuntos de entrenamiento y prueba, se aplican técnicas de escalado y se entrenan los modelos. Finalmente, se comparan las matrices de confusión y las métricas de rendimiento de cada modelo.

## Características principales del proyecto
-	Exploración de datos: Análisis exhaustivo de los datos, incluyendo la búsqueda de valores perdidos y duplicados.
-	Visualización de correlaciones: Uso de gráficos de pares y mapas de calor para identificar correlaciones entre las variables predictoras.
-	Selección de variables: Selección de las variables más relevantes para los modelos de aprendizaje automático.
-	Entrenamiento y prueba: División de los datos en conjuntos de entrenamiento y prueba, y aplicación de técnicas de escalado.
-	Comparación de modelos: Entrenamiento de cinco modelos de aprendizaje automático y comparación de sus matrices de confusión y métricas de rendimiento.
-	Análisis de AUC-ROC: Evaluación del rendimiento de los modelos mediante la curva AUC-ROC.
-	Visualización de métricas: Comparación de las métricas de precisión, sensibilidad, F1-score y exactitud mediante gráficos de barras.

## Herramientas utilizadas
**Lenguaje de programación**: 
-	Python
**Librerías de Python**:
-	Pandas (para manipulación de datos)
-	Numpy (para operaciones numéricas)
-	Matplotlib y seaborn (para visualización de datos)
-	Scikit-learn (para preprocesamiento y evaluación de modelos)
-	Entorno de desarrollo: Jupyter Notebook
-	Dataset: Wisconsin Breast Cancer Dataset

## Conclusión 
El resultado de las primeras comparaciones de los modelos de aprendizaje automático mostró que la regresión logística es el modelo más preciso y eficiente para la detección de cáncer de mama, con una precisión, sensibilidad y F1-score de 0.98. El bosque aleatorio también muestra un rendimiento excelente con 0.97 en estas métricas. Las máquinas de vectores de soporte tienen un buen rendimiento con 0.95, mientras que el árbol de decisión y el Bayesiano ingenuo gaussiano son menos precisos, con 0.91 en precisión y sensibilidad, pero pueden ser útiles en contextos específicos. Sin embargo, al analizar la curva AUC-ROC, se observa que el bosque aleatorio tiene el mayor AUC, lo que sugiere que es el mejor modelo para este conjunto de datos. Por lo tanto, se recomienda utilizar el bosque aleatorio como el modelo principal para optimizar la precisión en esta tarea de clasificación. Observando el gráfico de barras, se pueden comparar las métricas de cada modelo, y el bosque aleatorio muestra la mejor combinación de precisión, sensibilidad, F1-score y exactitud en este conjunto de datos.

## Inspiración y agradecimiento
Este proyecto se inspiró en este trabajo compartido en Kaggle Feature Selection and Data Visualization (kaggle.com), el cual ofreció valiosas percepciones sobre el aprendizaje. Agradezco enormemente al autor por compartir su conocimiento y recursos con la comunidad de código abierto.

## Estado del proyecto
Finalizado 

