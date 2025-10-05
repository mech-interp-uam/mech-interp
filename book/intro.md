# Introducción

En este proyecto nos propusimos recorrer el camino matemático que lleva

desde las redes neuronales más simples hasta la arquitectura Transformer,
destacando cómo cada avance ha ido construyendo las bases para el siguien-
te. Nuestros objetivos principales fueron, por un lado, mostrar la evolución
de estas estructuras y el papel central que desempeñan las matemáticas en su
funcionamiento interno; y, por otro, retomar el trabajo presentado en el pa-
per "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on
Gemma 2", aplicándolo sobre el modelo Llama 3.2 1B con el fin de explorar
cómo los autoencoders dispersos pueden ayudarnos a comprender mejor las
representaciones latentes de los modelos de lenguaje y de esta forma ver si
es posible una mejor interpretación dentro de lo que sucede en estos modelos
de lenguaje.

Comenzamos revisando los perceptrones y las primeras redes feedforward,
para después introducir arquitecturas más complejas. A lo largo de este re-
corrido, pusimos especial atención en los Autoencoders Dispersos (Sparse
Autoencoders, SAE), que permiten proyectar las activaciones internas hacia
representaciones dispersas y, por lo tanto, más interpretables.

Posteriormente, abordamos el Transformer, una arquitectura que mar-
có un cambio de paradigma al reemplazar la recurrencia por el mecanismo
de atención, cuya formulación matemática —basada en productos escalares,
normalizaciones y operaciones matriciales— constituye un gran avance en el
aprendizaje maquinal. Finalmente, realizamos una serie de pruebas inspira-
das en el enfoque de "Gemma Scope", con el objetivo de evaluar de manera
práctica la aplicabilidad de estas ideas en nuestro proyecto.




```{tableofcontents}
```

# Objetivo del libro

* Entender la arquitectura  de los modelos neuronales modernos
* Aplicar técnicas de ingeniería inversa a un modelo LLAMA
* Interpretar pesos y activaciones de modelos entrenados 

Este libro está dirigido a estudiantes, investigadores y entusiastas del aprendizaje automático interesados en la comprensión profunda del comportamiento interno de redes neuronales complejas.
