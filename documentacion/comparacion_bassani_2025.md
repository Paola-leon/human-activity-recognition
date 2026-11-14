# Comparación con Bassani et al. (2025)

**Referencia:** Bassani, G., Avizzano, C. A., y Filippeschi, A. (2025). *Deep Learning Algorithms for Human Activity Recognition in Manual Material Handling Tasks*. Sensors, 25, 6705. https://doi.org/10.3390/s25216705

El artículo usa el mismo conjunto de datos que nuestro proyecto: el MMH de Zenodo, con 14 sujetos y 7 actividades. Este documento reúne:

1. cómo se diferencian sus decisiones de las nuestras;
2. lo que su texto no especifica;
3. las inconsistencias internas que encontramos;
4. con qué métricas suyas sí se pueden comparar nuestros resultados.

Las referencias a secciones, tablas y figuras son las del artículo.

---

## 1. Diseño experimental: ellos frente a nosotros

| Aspecto | Bassani et al. (2025) | Nuestro proyecto |
|---|---|---|
| Sujetos usados | los 14 | 1–12 en la validación cruzada; 13 y 14 reservados para la prueba final |
| Frecuencia | 240 Hz (entrada de la RCNN `10 × 240`, sección 3.2.4) | 12 Hz |
| Ventana | 1 s, paso de 1 s, sin traslape (sección 3.1) | 4 s (48 pasos), una ventana etiquetada cada 0.5 s |
| Etiqueta de la ventana | **no se especifica** | actividad del **instante final** de la ventana |
| Entrada | 10 señales: 8 de movimiento + 2 de sEMG (bíceps y braquiorradial) | `EML`: 10 variables de movimiento de 11 sensores (codos, muñecas, caderas, rodillas, rapidez de manos); sin sEMG |
| Balanceo de clases | submuestreo de N y réplica de las clases minoritarias con ruido gaussiano (sección 3.1) | pesos de clase en la pérdida; sin réplicas |
| Protocolo principal | partición aleatoria 70–30 de las ventanas | LOSO con 12 rondas |
| "LOSO" | entrenar con los sujetos 1–13 y probar en el **sujeto 14** (sección 3.3) | 12 rondas: cada sujeto se evalúa una vez; 10 entrenan y 1 es la validación interna |
| Selección de hiperparámetros | búsqueda en malla con la partición 70–30 de los 14 sujetos (sección 3.3) | validación interna con un sujeto que no se evalúa |
| Modelos | BiLSTM, Sp-DAE, Sp-DAE recurrente, RCNN; DeepConvLSTM como referencia | CPC-GRU (preentrenamiento sin etiquetas + LP-FT) y Random Forest |
| Métrica | F1 macro (sección 3.3, *Metrics*) | F1 macro por sujeto, con media, desviación y peor sujeto |

---

## 2. Hallazgos sobre su metodología

### 2.1 No se especifica cómo se etiqueta cada ventana

Todo lo que dice el texto (sección 3.1, p. 6) es:

> "the time series are segmented by a sliding window of a fixed length […] Ideally, each segment should contain one activity […] we found a good compromise using short intervals of 1 s length with 1 s stride and no overlap."

- **"Idealmente cada segmento contiene una actividad"** es una aspiración, no una regla.
- **No se dice qué etiqueta reciben las ventanas que cruzan una transición.** Podría ser la actividad mayoritaria, la del instante final o la del centro, o esas ventanas podrían descartarse.
- **La sección 3.2.1 solo indica** que la capa de clasificación usa entropía cruzada "for mutually exclusive classes", es decir una etiqueta por ventana.

**Por qué importa:** con nuestros datos, una parte de las ventanas de 4 s cruza una transición entre actividades y se etiqueta con el último instante; esas ventanas son ambiguas por construcción (notebook 16, sección 9.2). Si ellos descartaron las ventanas con transición, su tarea sería más fácil que la nuestra.

### 2.2 El balanceo se hizo antes de dividir en entrenamiento y prueba

El texto no dice si el submuestreo de N y la réplica con ruido se aplicaron antes o después de la partición 70–30. La **figura 8** (matriz de confusión del BiLSTM en el conjunto de prueba 70–30) permite deducirlo. Sumando por clase verdadera:

| Clase verdadera | N | LT | PT | LF | PF | K | W | Total |
|---|---|---|---|---|---|---|---|---|
| Ventanas de prueba | 1,637 | 1,389 | 1,092 | 1,305 | 1,385 | 1,774 | 1,529 | 10,111 |

- **N tiene un tamaño similar al de las demás clases,** aunque los autores la describen como la clase más grande (sección 3.1).
- **Si el balanceo se hubiera aplicado solo al entrenamiento,** la prueba conservaría la distribución natural, con N dominando.
- **Conclusión:** el balanceo se hizo **antes** de dividir. El 30 % de prueba incluye **copias con ruido de ventanas que también están en entrenamiento**.

Es una fuga más fuerte que la que medimos nosotros en una partición 70–30. En el notebook 16, el 90.9 % de las ventanas de prueba tenía en entrenamiento una vecina que comparte el 87.5 % de sus instantes, y aun así el F1 solo subió ≈ 0.03 frente a LOSO.

### 2.3 Los hiperparámetros se eligieron con el mismo conjunto en que se reportan

- **Cómo se eligieron:** la sección 3.3 describe una búsqueda en malla de épocas y unidades ocultas (figuras 2–5) "optimizing the classification performance" con la partición 70–30 de los 14 sujetos. No se menciona un conjunto de validación separado.
- **Qué se reporta:** el 95.7 % del BiLSTM es el **máximo** de 50 configuraciones (sección 4.1.1) evaluadas en ese mismo 30 %.
- **Consecuencia:** el valor es optimista por selección, además de por la fuga del punto 2.2.

### 2.4 Su "LOSO" es un solo sujeto

- **Cómo lo hacen:** entrenan con 1 a 13 sujetos y prueban siempre en el **sujeto 14** (sección 3.3 y figura 7).
- **Qué significa:** el 90.3 % es el F1 de **una persona**, no un promedio sobre rotaciones.
- **Por qué importa:** en nuestro LOSO, el F1 por sujeto varía mucho. Con CPC-GRU (`EML`) va de 0.775 (sujeto 2) a 0.908 (sujeto 4). Un solo sujeto puede caer en cualquier parte de ese rango.

### 2.5 Coincidencia con nuestros resultados

La sección 5 y la figura 8 señalan que **PT es la actividad más difícil** y que se confunde sobre todo con N, "maybe due to its little variation in joint flexions, especially when handling light loads". Es la misma confusión y la misma explicación que encontramos en el notebook 14 (sección 7.3 del [resumen](resumen_notebook_14_cabeza_lineal_vs_mlp.md)).

---

## 3. Inconsistencias del artículo

| # | Dónde | Qué dice | Por qué es inconsistente |
|---|---|---|---|
| 1 | Resumen, sección 4.3 y figura 7 frente a tabla 5 | F1 LOSO del BiLSTM: **90.3 %** en el resumen y en la sección 4.3; **90.6 %** en la tabla 5 | Es el mismo modelo y el mismo protocolo con dos valores distintos |
| 2 | Tabla 5 frente a tabla 4 | Exactitud LOSO de **97.2 %** (BiLSTM) y **96.9 %** (RCNN); F1 por clase en LOSO de **84.0–94.5 %** (BiLSTM) y **85.1–92.4 %** (RCNN) | Con una etiqueta por ventana, la exactitud es igual al F1 micro, y el F1 micro no puede superar al F1 de la mejor clase. Con los valores de la tabla 4, la exactitud LOSO no podría pasar de 94.5 % (BiLSTM) ni de 92.4 % (RCNN) |
| 3 | Tabla 5 | Exactitud LOSO (97.2 %) **mayor** que la exactitud 70–30 (96.0 %), mientras que el F1 LOSO (90.6 %) es mucho **menor** que el F1 70–30 (95.7 %) | Las dos métricas se mueven en sentidos opuestos entre protocolos. Es otro indicio de que la exactitud LOSO está mal calculada o se calculó sobre un conjunto distinto |
| 4 | Tabla 1 | Tasa de aprendizaje inicial "1 × 10³" | Probablemente es 1 × 10⁻³: una tasa de 1000 no permitiría entrenar |
| 5 | Sección 3.3 y figura 7 | Llaman "LOSO" a una sola partición (sujetos 1–13 → sujeto 14) | En LOSO cada sujeto se deja fuera una vez y se promedia. Con un solo sujeto de prueba, no hay rotación |
| 6 | Pie de la figura 7 | Menciona "(a) BiLSTM … and (b) RCNN" | La figura tiene un solo panel con las dos curvas |
| 7 | Sección 3.1 frente a figura 8 | No se indica si el balanceo es antes o después de dividir | La distribución de clases de la prueba en la figura 8 muestra que fue antes (punto 2.2) |

---

## 4. Con qué métricas sí podemos compararnos

### 4.1 Comparables (con las precauciones indicadas)

**a) F1 macro en un sujeto nuevo (tabla 5 y figura 7).**

| Modelo | F1 en sujeto nuevo | Protocolo |
|---|---|---|
| BiLSTM (Bassani) | 0.903 (o 0.906, ver inconsistencia 1) | 13 sujetos → sujeto 14 |
| RCNN (Bassani) | 0.892 | 13 sujetos → sujeto 14 |
| DeepConvLSTM (Bassani) | 0.903 | 13 sujetos → sujeto 14 |
| CPC-GRU, `EML` (nuestro) | 0.873 ± 0.036 (rango 0.775–0.908) | LOSO, 12 rondas, 10 sujetos de entrenamiento |
| Random Forest, 30 variables (nuestro) | 0.894 | LOSO, 12 rondas |
| Random Forest, 30 variables + sEMG, ventanas de 1 s (nuestro, notebook 16 parte B) | 0.908 | LOSO, 12 rondas |

**Precauciones:**
- **Su valor es de un solo sujeto con 13 de entrenamiento;** el nuestro es el promedio de 12 sujetos con 10 de entrenamiento.
- **Sus 0.892–0.906 caen dentro de nuestro rango por sujeto.**
- **La comparación más justa** sería reportar nuestro modelo en el **sujeto 14**, que está reservado para la prueba final. Recomendamos reportarlo por separado cuando se haga esa evaluación.

**b) F1 por actividad en un sujeto nuevo (tabla 4).**

| Actividad | BiLSTM (Bassani, sujeto 14) | RCNN (Bassani, sujeto 14) | CPC-GRU `EML` (nuestro, 12 sujetos) |
|---|---|---|---|
| N | 0.891 | 0.901 | 0.917 |
| LT | 0.945 | 0.924 | 0.823 |
| PT | 0.840 | 0.851 | 0.821 |
| LF | 0.903 | 0.887 | 0.837 |
| PF | 0.888 | 0.867 | 0.889 |
| K | 0.923 | 0.914 | 0.914 |
| W | 0.934 | 0.902 | 0.933 |

- **En los dos casos PT es de las más difíciles.**
- **La diferencia más grande está en LT y LF,** donde ellos tienen la sEMG del brazo (información de carga) y ventanas de 1 s.
- **Nuestro Random Forest de la parte B** (con sEMG y ventanas de 1 s) sube LF a 0.899, LT a 0.878 y PT a 0.865, más cerca de sus valores.

**c) F1 frente al número de sujetos de entrenamiento (figura 7).**

| Sujetos de entrenamiento | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| BiLSTM (Bassani, sujeto 14; ≈ leído de la figura 7) | ≈ 0.80 | ≈ 0.84 | ≈ 0.88 | ≈ 0.90 |
| CPC-GRU con sus 10 variables (nuestro, notebook 12) | 0.826 | 0.852 | 0.868 | 0.880 |

- **Las dos curvas siguen creciendo con más sujetos.**
- **La comparación es de tendencia, no de valores exactos:** los suyos se leyeron de la gráfica y son de un solo sujeto de prueba.

**d) Confusión PT ↔ N (sección 5, figura 8).** Es una comparación cualitativa, y coincide con la nuestra.

**e) Número de parámetros entrenables (tabla 5).**

| Modelo | Parámetros |
|---|---|
| BiLSTM (Bassani) | 750,607 |
| RCNN (Bassani) | 756,675 |
| DeepConvLSTM (Bassani) | 15,989,191 |
| CPC-GRU con cabeza lineal (nuestro) | 725,415 |

Nuestro modelo final tiene un tamaño equivalente al de sus redes ligeras.

### 4.2 No comparables

| Métrica | Dónde | Por qué no |
|---|---|---|
| F1, exactitud y precisión 70–30 (95.7 %, 95.9 %, 95.2 %) | tabla 5, figuras 2–6 y 8 | ventanas de los mismos sujetos en entrenamiento y prueba, réplicas con ruido en ambos lados (punto 2.2) e hiperparámetros elegidos con ese mismo conjunto (punto 2.3) |
| Exactitud LOSO (97.2 %, 96.9 %) | tabla 5 | inconsistente con su propio F1 por clase (inconsistencia 2) |
| F1 frente al número de sujetos en 70–30 | figura 6 | mide reconocer ventanas de personas ya vistas, no personas nuevas |
| MAC, MA, memoria y latencia | tabla 5 | no los hemos medido; se podrían calcular para nuestro modelo si se quiere comparar la complejidad |
| Desempeño con pocas etiquetas | — | el artículo no lo evalúa; es la contribución propia de nuestro proyecto (notebook 17) |

---

## 5. Resumen

- **El 95.7 % del artículo no es comparable con nuestro LOSO:**
  - viene de una partición 70–30 con réplicas con ruido en ambos lados;
  - los hiperparámetros se eligieron con ese mismo conjunto;
  - no se especifica cómo se etiquetan las ventanas.
- **La cifra comparable es su F1 en sujeto nuevo (0.903).** Es un solo sujeto, y cae dentro de nuestro rango por sujeto. Nuestro mejor LOSO, el Random Forest con 30 variables + sEMG (0.908 de media en 12 sujetos), ya está a ese nivel.
- **Las dos líneas de trabajo coinciden en tres cosas:** PT es la actividad más difícil, su confusión principal es con N, y el F1 sigue subiendo con más sujetos de entrenamiento.
- **Lo que el artículo no aborda:** la dependencia de las etiquetas. Es donde CPC muestra su ventaja (notebook 17).
