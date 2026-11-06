# BiLSTM y xLSTM como modelos autorregresivos de CPC: por qué no son prioridad

**Proyecto Integrador · Equipo 1** · 5 de noviembre de 2026

## Resumen

- **BiLSTM:** no se puede usar como modelo autorregresivo (`g_ar`) de CPC tal cual. Su lectura hacia atrás ya conoce el futuro que CPC debe predecir, así que el preentrenamiento deja de ser válido. Como clasificador entrenado desde cero sí es válido; esa prueba está pendiente.
- **xLSTM:** sí es compatible con CPC, y el proyecto ya probó sus dos tipos de celda. La sLSTM quedó en 0.875 y la mLSTM en 0.862, ambas desde cero. Las diferencias con GRU y LSTM son menores que la variación entre sujetos y entre semillas, así que un xLSTM completo difícilmente aportaría una mejora clara para su costo.
- **Random Forest frente a CPC:** el Random Forest gana sólo con todas las etiquetas (0.894 contra 0.873 del ensamble CPC-GRU). En el escenario real, con pocas etiquetas, el ensamble CPC lo supera con 1 a 100 ventanas por actividad, hasta +0.071 con 10. Es lo que predicen van den Oord et al. (2018) y Haresamudram et al. (2020). La confirmación final queda para los sujetos 13 y 14, que siguen sin usarse (sección 4).
- **Alcance:** la parte del BiLSTM es un argumento teórico, no medido con nuestros datos. La parte de xLSTM se basa en resultados de una sola semilla y en un diagnóstico de estabilidad con un solo sujeto.

---

## 1. BiLSTM como `g_ar` de CPC

### Cómo funciona un BiLSTM

Un BiLSTM son dos LSTM que leen la misma secuencia en sentidos opuestos:

- **Hacia adelante:** lee de `z1` a `zT`. En cada instante `t`, resume lo ocurrido hasta `t` (el pasado).
- **Hacia atrás:** lee de `zT` a `z1`. En cada instante `t`, resume lo que ocurre desde `t` hasta el final (el futuro).

En cada instante se concatenan los dos estados, así que la salida en `t` tiene información de toda la secuencia.

```text
adelante   z1 ═══════════► t            (ve el pasado)
atrás                      t ◄═══════════ zT   (ve el futuro)
```

### Por qué no sirve dentro de CPC

CPC aprende con una tarea concreta: **a partir del contexto `c_t`, reconocer el futuro real `z(t+k)` entre varios candidatos** (pérdida InfoNCE). La tarea sólo funciona si `c_t` no conoce el futuro.

1. **El modelo ya tiene la respuesta.** Con un BiLSTM, `c_t` incluye la lectura hacia atrás, que ya pasó por `z(t+k)`. La tarea deja de ser predecir y se vuelve identificar algo que el modelo ya vio.
2. **La representación pierde utilidad.** CPC obliga a captar patrones del movimiento que permitan anticipar lo que sigue. Si el modelo puede reconocer el futuro directamente, desaparece esa presión, y la representación puede servir poco para clasificar actividades.
3. **La pérdida se vuelve engañosa.** InfoNCE podría bajar mucho y la precisión contrastiva subir sin que el modelo aprenda nada útil. Esas métricas parecerían mejores que las de GRU, LSTM o sLSTM, pero no serían comparables.
4. **Rompe un requisito del pipeline.** El notebook 10 verifica que cambiar la señal a partir de `T/2` no altere nada antes de `T/2`. Un BiLSTM falla esa verificación por diseño.

Un detalle práctico adicional: el notebook 10 representa cada ventana con el último paso de la secuencia (`c_T`). En un BiLSTM, ese paso de la lectura hacia atrás sólo ha visto un instante. Si se usa como clasificador, hay que tomar el estado final de cada dirección: la lectura hacia adelante en `t = T` y la inversa en `t = 1`.

### Qué no es un problema

- **Clasificar con un BiLSTM es válido.** La etiqueta de cada ventana es la de su último instante, así que todo lo que lee el BiLSTM es pasado respecto a ese instante. No hay fuga de información ni espera adicional en tiempo real.
- **El problema está sólo en el preentrenamiento.** Por eso, la prueba propuesta es un BiLSTM entrenado desde cero, sin CPC, comparado con la LSTM bajo el mismo protocolo.

### Si se quisiera usar con CPC

Habría que preentrenar **dos LSTM por separado** sobre el mismo encoder: una ve el pasado y predice el futuro, y otra ve el futuro y predice el pasado, cada una con su propio predictor y su propia pérdida. Sus salidas se concatenan sólo al clasificar. Sería una variante del equipo, no el CPC original, y sólo vale la pena construirla si el BiLSTM desde cero iguala o mejora a la LSTM.

### Estado de la prueba

- **Pendiente:** no hay resultados del BiLSTM.
- **Por qué no se corrió todavía:** al preparar la comparación, la LSTM de referencia resultó inestable (sección 3), así que comparar con una sola semilla no sería confiable.
- **Propuesta:** correr LSTM y BiLSTM desde cero con 3 semillas, durante 30 épocas completas y sin parada temprana. La mejor época se elegiría con el sujeto de validación interna.

---

## 2. xLSTM como `g_ar`

### Qué es xLSTM

xLSTM (Beck et al., 2024) es una familia de redes, no un modelo único. Sus dos tipos de celda comparten una **compuerta de entrada exponencial**, con normalización y estabilización, que permite corregir con fuerza lo que hay en memoria.

| | sLSTM (*scalar*) | mLSTM (*matrix*) |
|---|---|---|
| **Memoria** | Un valor por celda, como la LSTM | Una matriz clave → valor por *head* |
| **Qué aporta** | Comparte estado entre celdas (*memory mixing*) y normaliza la memoria | Más capacidad de memoria, con una lectura parecida a la atención |
| **Cómo procesa** | Paso a paso, sin paralelizar | Toda la ventana en paralelo, sin recurrencia oculta |
| **Costo principal** | Lentitud por el recorrido paso a paso | Más memoria; nuestra versión usa una matriz de 48 × 48 por ventana |

Un xLSTM completo apila bloques de ambos tipos en una misma red, con conexiones residuales.

### Lo que el proyecto ya probó

- **sLSTM:** es el `g_ar` xLSTM del proyecto desde el notebook 06, y forma parte de la comparación del notebook 10.
- **mLSTM:** se comparó con la sLSTM y la GRU en el notebook 06, con sEMG, y quedó por debajo. Con MVNX se entrenó desde cero en el avance 4.
- **Nunca se ha validado:** la mLSTM con CPC y MVNX, y un xLSTM con bloques de ambos tipos.

Esta es la mejor variante de cada modelo con MVNX, según el avance 4 (LOSO, 12 sujetos, una semilla):

| Modelo | F1 macro | Desviación entre sujetos | Peor sujeto | Min por ronda |
|---|---|---|---|---|
| Random Forest | 0.894 | 0.026 | 0.823 | 0.2 |
| GRU con CPC | 0.878 | 0.020 | 0.835 | 5.0 |
| sLSTM desde cero | 0.875 | 0.028 | 0.816 | 10.1 |
| mLSTM desde cero | 0.862 | 0.041 | 0.775 | 12.6 |
| LSTM desde cero | 0.854 | 0.032 | 0.787 | 3.9 |

No todos los modelos usaron el mismo protocolo de ajuste fino. Además, los tiempos se midieron en equipos distintos, así que son orientativos.

### Por qué no se espera una mejora mucho mayor

1. **Las redes ya están muy parejas.** Entre la GRU, la sLSTM y la mLSTM hay como mucho 0.016 de F1, menos que la desviación entre sujetos (0.020–0.041). Con una sola semilla, esas diferencias no permiten elegir una celda.
2. **El ruido entre semillas es mayor que esas diferencias.** En la LSTM desde cero, el mismo sujeto dio entre 0.61 y 0.86 según la semilla (sección 3). Un xLSTM tendría que superar ese ruido para mostrar una mejora real.
3. **Las ventanas son cortas.** La memoria matricial está pensada para recordar más información en secuencias largas, y las ventanas del proyecto duran 4 s (48 pasos). Es una interpretación del equipo; no se ha probado con ventanas más largas.
4. **El límite parece estar en otro lado.** El Random Forest obtiene el mejor F1 con estadísticas simples por ventana, sin modelar el orden temporal. Eso sugiere que la dificultad principal está en las diferencias entre personas, y una celda recurrente más expresiva no ataca ese problema directamente.
5. **El costo crece.** La sLSTM y la mLSTM tardan unas 3 veces más que la LSTM desde cero, y la sLSTM con CPC llegó a 24 min por ronda, contra 5.4 de la LSTM. Un xLSTM completo tomaría unos 10–13 min por ronda desde cero, unas 2.5 h por semilla en los 12 sujetos. Es una estimación, no una medición.

### Escenarios en que sí podría aportar

En orden de prioridad:

1. **mLSTM en un ensamble.** El mejor resultado del avance 5 fue un ensamble de semillas de CPC-GRU. Una celda muy distinta podría cometer errores diferentes y sumar al ensamble.
2. **mLSTM con CPC.** Es el hueco más claro de la comparación con MVNX, y sería relevante para el escenario de pocas etiquetas, que es la razón por la que se eligió el modelo final.
3. **Ventanas más largas.** Ahí la memoria podría empezar a importar, aunque habría que rehacer la comparación.
4. **xLSTM completo.** Es lo más costoso, y no se sabe si aporta más que un ensamble.

---

## 3. Hallazgo relacionado: inestabilidad de la LSTM desde cero

Al preparar la comparación del BiLSTM, se repitió la LSTM desde cero del notebook 10 en el sujeto 1, con el mismo protocolo LP-FT (hasta 30 épocas, paciencia 5).

| Corrida | Épocas | F1 en el sujeto evaluado |
|---|---|---|
| Notebook 10, semilla 0 | — | 0.815 |
| Semilla 0, con parada temprana (dos repeticiones) | 16 | 0.632 y 0.608 |
| Semilla 1, con parada temprana | 30 | 0.839 |
| Semilla 2, con parada temprana | 27 | 0.857 |
| Semilla 0, sin parada temprana (dos repeticiones) | 30 | 0.806 y 0.812 |

- **La parada temprana es la causa principal.** La LSTM se estanca varias épocas y luego mejora tarde. Con la semilla 0, la corrida se detuvo en la época 16; en la 17 la validación ya habría mejorado. Sin parada temprana llega a ≈ 0.81, como en el notebook 10.
- **La LSTM sigue mejorando en la época 30.** En ambas repeticiones sin parada temprana, la mejor validación fue la última época.
- **Hay algo de ruido incluso con la misma semilla.** Las repeticiones dentro de un mismo proceso no coinciden exactamente (0.632 y 0.608), aunque esa diferencia es mucho menor que la que hay entre semillas.
- **Implicaciones:** el 0.854 de la LSTM desde cero, de una sola semilla, podría estar subestimado. Las comparaciones del notebook 10 que la involucran deben leerse con cautela, como preentrenado contra desde cero. No se ha revisado si la GRU y la sLSTM tienen el mismo problema.
- **Limitación:** el diagnóstico se hizo con un solo sujeto, y mirar ese sujeto influyó en la propuesta de quitar la parada temprana. Como el cambio se aplica por igual a todos los sujetos y modelos, la contaminación es menor, pero se deja documentada. Los sujetos 13 y 14 no se usaron.
- **Una hipótesis sin probar:** la sLSTM del proyecto empieza con un sesgo de la compuerta de olvido que favorece recordar (entre 3 y 6), y `nn.LSTM` de PyTorch no. Eso podría explicar parte del estancamiento inicial de la LSTM.

---

## 4. Random Forest frente a CPC: el aporte está en el escenario con pocas etiquetas

**Tesis.** El Random Forest gana sólo en el escenario de laboratorio, donde todas las ventanas tienen etiqueta: 0.894 de F1 macro, contra 0.873 del ensamble de semillas CPC-GRU (el modelo final) y 0.878 de la GRU con CPC y ajuste fino. En el escenario real, con pocas etiquetas, el orden se invierte: con 1 a 100 ventanas etiquetadas por actividad, el ensamble CPC supera al Random Forest en todos los tamaños probados (avance 5, sección 11.4). Es lo que predicen los dos papers de referencia de CPC (van den Oord et al., 2018; Haresamudram et al., 2020). La confirmación definitiva queda para los sujetos 13 y 14, que siguen sin usarse.

### 4.1 Laboratorio contra planta real

| | Laboratorio (LOSO actual) | Planta real |
|---|---|---|
| **Grabar movimiento** | Sesiones controladas | Barato: el traje graba horas de trabajo |
| **Etiquetar** | Todas las ventanas, anotadas cuadro a cuadro | Caro: alguien tiene que revisar el video y anotar cada actividad |
| **Etiquetas disponibles** | Todas | Unas decenas de ejemplos por actividad |
| **Qué modelo conviene** | Random Forest, por poco (+0.021) | CPC: sólo el clasificador lineal necesita etiquetas |

Haresamudram et al. (2020) parten justo de esta asimetría: en HAR con sensores corporales, grabar datos es sencillo, pero conseguir sus anotaciones no lo es. Por eso proponen CPC para "hacer un uso económico de las pocas etiquetas y aprovechar los datos sin etiquetar". Nuestro LOSO con todas las etiquetas mide el caso más favorable para un modelo supervisado, no el caso de uso.

### 4.2 Qué dicen los papers con todas las etiquetas

| Fuente | Tarea | Supervisado | CPC congelado + clasificador | Diferencia |
|---|---|---|---|---|
| van den Oord et al., LibriSpeech | Fonemas (41 clases) | 74.6 | 64.6 (lineal) · 72.5 (una capa oculta) | −10.0 · −2.1 |
| van den Oord et al., LibriSpeech | Hablante (251 clases) | 98.5 | 97.4 (lineal) | −1.1 |
| Haresamudram et al., Mobiact | HAR, 11 clases | 83.68 | 80.97 (MLP) | −2.7 |
| Haresamudram et al., Motionsense | HAR, 6 clases | 86.66 | 89.05 (MLP) | +2.4 |
| Haresamudram et al., UCI-HAR | HAR, 6 clases | 79.79 | 81.65 (MLP) | +1.9 |
| Haresamudram et al., USC-HAD | HAR, 12 clases | 49.09 | 52.01 (MLP) | +2.9 |
| **Este proyecto, MVNX** | HAR, 7 clases | **0.894 (RF)** | 0.873 (ensamble lineal) · 0.878 (ajuste fino) | −0.021 · −0.016 |

En van den Oord se reporta exactitud; en Haresamudram y en este proyecto, F1 macro. En Haresamudram, "supervisado" es la misma arquitectura de CPC entrenada de extremo a extremo con todas las etiquetas.¹

Con todas las etiquetas, CPC queda entre 3 puntos abajo y 3 arriba del modelo supervisado en HAR, y por debajo en el paper original. **Ninguno de los dos papers espera que CPC gane con todas las etiquetas.** Nuestra diferencia (−0.016 a −0.021) cae dentro de ese rango y es menor que la desviación entre sujetos (0.020–0.026).

### 4.3 Por qué el Random Forest gana con todas las etiquetas

1. **CPC no optimiza la clasificación.** InfoNCE maximiza una cota inferior de la información mutua entre el contexto y el futuro: I(x(t+k); c_t) ≥ log N − L_N (van den Oord et al., 2018). La representación guarda lo que sirve para anticipar el movimiento, no necesariamente lo que separa las siete actividades. Con todas las etiquetas, un modelo entrenado directamente para la tarea, como el Random Forest, aprovecha esa ventaja.
2. **En el LOSO, CPC no ve datos que el Random Forest no tenga.** En los papers, CPC se preentrena con mucho más dato del que se etiqueta: 100 h de habla, o el dataset completo para clasificar después con pocas ventanas. Haresamudram et al. observan que el dataset más grande (Mobiact, 61 personas) es el que más gana con el preentrenamiento. En nuestro LOSO, CPC se preentrena con las mismas ventanas de los mismos 10 sujetos que después tienen etiqueta, así que no aporta información nueva. En una planta, en cambio, las horas sin etiquetar serían justamente lo que CPC aprovecha.
3. **La entrada ya está diseñada a mano y tiene pocas dimensiones.** CPC se pensó para datos crudos de muchas dimensiones, como audio a 16 kHz, donde supera a los MFCC por 25 puntos (64.6 contra 39.7). Nuestras 30 variables MVNX a 12 Hz ya describen postura, alcance, velocidad y apoyo, y son invariantes a la posición. Eso hace buena parte del trabajo de representación, y Haresamudram et al. recuerdan que las características estadísticas siguen siendo una línea base fuerte en HAR.
4. **Las actividades se distinguen sobre todo por postura.** CPC capta "características lentas" que abarcan muchos pasos de tiempo. Aquí lo decisivo es dónde está el cuerpo (altura de las manos, flexión del tronco), y la media, el mínimo y el máximo lo capturan directamente. Es una interpretación del equipo, no se midió.
5. **Clasificador lineal contra no lineal.** El modelo final usa regresión logística, y el Random Forest usa 300 árboles. En van den Oord, una capa oculta sube los fonemas de 64.6 a 72.5 sobre las mismas características: "no toda la información codificada es accesible linealmente". Con el mismo clasificador lineal, la representación de CPC (0.820) supera ampliamente a las medias de las 30 variables (0.615).

### 4.4 Evidencia del proyecto: con pocas etiquetas, CPC supera al Random Forest

En cada ronda LOSO se entrenó el clasificador con sólo *n* ventanas etiquetadas por actividad, tomadas al azar de los sujetos de entrenamiento (3 repeticiones por tamaño). Los CPC no cambian, porque se preentrenaron sin etiquetas, y el Random Forest se entrena con las mismas ventanas. Los valores son el F1 macro medio de los 12 sujetos evaluados (avance 5, sección 11.4).

| Ventanas etiquetadas por actividad | Ensamble CPC-GRU | Sonda CPC-sLSTM | Random Forest | Ensamble − RF |
|---|---|---|---|---|
| 1 (7 en total) | 0.361 | 0.353 | 0.337 | +0.024 |
| 5 (35) | **0.615** | 0.565 | 0.559 | **+0.056** |
| 10 (70) | **0.691** | 0.640 | 0.620 | **+0.071** |
| 25 (175) | **0.771** | 0.714 | 0.729 | **+0.042** |
| 50 (350) | **0.805** | 0.745 | 0.782 | **+0.023** |
| 100 (700) | 0.828 | 0.767 | 0.824 | +0.004 |
| Todas | 0.873 | 0.813 | **0.894** | −0.021 |

- **La ventaja de CPC crece cuando las etiquetas escasean.** Llega a su máximo con 10 ventanas por actividad (+0.071) y desaparece entre 100 y el conjunto completo. Es el mismo patrón que la figura 2 de Haresamudram et al.: con 1 a 100 ventanas por clase, CPC supera a DeepConvLSTM por 5 a más de 10 puntos en los cuatro datasets, con una sola excepción (Motionsense con una ventana por clase).
- **Con 50 ventanas por actividad** (350 en total; a lo sumo ≈ 23 min de movimiento anotado), el ensamble alcanza 0.805, el 92 % de su F1 con todas las etiquetas. Para llegar ahí, el Random Forest necesita entre 50 y 100.
- **El rival es más fuerte que en el paper.** Haresamudram et al. comparan contra una red profunda (DeepConvLSTM), que sufre mucho con pocos datos. Un Random Forest sobre 150 estadísticas es un rival más duro en ese régimen, y aun así CPC lo supera de 5 a 50 ventanas por actividad.
- **Con una sola ventana por actividad, ningún modelo es útil** (F1 ≈ 0.34–0.36). La ventaja de +0.024 en ese tamaño no tiene valor práctico.

Por eso el modelo final es el ensamble de semillas CPC-GRU y no el Random Forest. Pierde 0.021 en el caso más favorable para el modelo supervisado, y gana hasta 0.071 en el caso de uso. Además, su estabilidad entre personas es igual o mejor: peor sujeto 0.827 contra 0.823, desviación 0.021 contra 0.026. Esto coincide con la observación de Haresamudram et al. de que el preentrenamiento hace el reconocimiento más robusto a la variabilidad de las ventanas.

### 4.5 Evaluación final con los sujetos 13 y 14

Los sujetos 13 y 14 no se han usado en ningún experimento: ni para preentrenar, ni para elegir hiperparámetros, ni para elegir el modelo final. Son la única prueba que no está influida por las decisiones del proyecto, mientras que en el LOSO los sujetos 11 y 12 sí influyeron: con ellos se eligieron la configuración de CPC (notebook 06) y los hiperparámetros del Random Forest y la GRU. Se propone usarlos así:

1. **Congelar todo antes de abrirlos.** Modelo final (ensamble de 5 semillas CPC-GRU con sonda lineal), Random Forest con su configuración original, los tamaños de etiquetas (1, 5, 10, 25, 50, 100 y todas) y las semillas de muestreo. Se entrena con los sujetos 1–12 y no se ajusta nada después de ver los resultados.
2. **Medir los dos escenarios.** Con todas las etiquetas, para comparar con el 0.894 y el 0.873 del LOSO; y la curva con pocas etiquetas, que es la prueba principal del aporte.
3. **Fijar de antemano qué cuenta como confirmación.** Se confirma la tesis si el ensamble supera al Random Forest con 5–50 ventanas por actividad en ambos sujetos. Si el Random Forest lo iguala o supera con pocas etiquetas, la ventaja del LOSO no se sostiene en personas nuevas y debe reportarse así.
4. **Opcional: el caso de planta, separado.** Preentrenar CPC también con el movimiento sin etiquetas de los sujetos 13 y 14 imitaría lo que pasaría en una planta, donde se graba a los trabajadores antes de etiquetarlos. Sería un resultado distinto, no un reemplazo de la evaluación anterior, y habría que decidirlo antes de abrir los datos.

Con dos sujetos no hay potencia estadística para afirmar una diferencia. La evaluación sirve para verificar que el patrón del LOSO se mantiene en personas nuevas, no para estimarlo con precisión.

### 4.6 Limitaciones

- **Ruido de la curva:** se basa en 3 repeticiones por tamaño, y las ventajas de +0.004 (100 ventanas) y +0.024 (1 ventana) son del orden del ruido.
- **Etiquetas de laboratorio:** las ventanas etiquetadas se toman al azar de los 10–11 sujetos de entrenamiento. En una planta, las pocas etiquetas vendrían quizá de una sola persona o sesión, lo que podría ser más difícil para ambos modelos.
- **Mismos datos sin etiquetar:** como en el LOSO, CPC se preentrenó sólo con los sujetos de entrenamiento. La ventaja medida no incluye el beneficio de tener más horas sin etiquetar, que es el principal argumento de los papers.
- **Cifras con todas las etiquetas de una sola corrida LOSO:** la diferencia con el Random Forest (0.021) es menor que la desviación entre sujetos.
- **Analogía con los papers:** ningún paper de CPC compara contra Random Forest. Las razones 3 y 4 de la sección 4.3 son interpretaciones del equipo.

¹ *En el texto de Haresamudram et al. se dice que CPC queda por debajo del entrenamiento de extremo a extremo en USC-HAD, pero su tabla 2 muestra lo contrario (52.01 contra 49.09). Aquí se usan las cifras de la tabla.*

---

## Referencias

- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://arxiv.org/abs/2405.04517
- Haresamudram, H., Essa, I., & Plötz, T. (2020). Contrastive predictive coding for human activity recognition. *arXiv:2012.05333*. https://arxiv.org/abs/2012.05333
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748*. https://arxiv.org/abs/1807.03748
