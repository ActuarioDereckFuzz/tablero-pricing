# Presentación final — PurePremium Lab

**Duración:** 10 minutos  
**Equipo:** 7 integrantes  
**Herramienta:** Aplicación Streamlit en vivo  
**Objetivo:** Comunicar resultados actuariales mediante una herramienta interactiva.

---

## Integrante 1 — Introducción
**Tiempo: 0:00–0:55**

**Pantalla:** Inicio de la aplicación o Metodología.

Buenos días.

Nuestro proyecto se llama **PurePremium Lab** y consiste en una aplicación interactiva para analizar la tarificación de seguros de automóviles utilizando modelos estadísticos, Machine Learning y Deep Learning.

En seguros, una de las principales tareas actuariales es estimar cuánto esperamos pagar en siniestros por un riesgo determinado; a esto lo conocemos como prima pura.

Sin embargo, estimar correctamente el riesgo no garantiza que una tarifa sea comercialmente conveniente.

También debemos considerar cuánto cobramos, qué clientes aceptan ese precio y cuál es la rentabilidad esperada de la cartera.

Por ello desarrollamos una herramienta en Streamlit que nos permite relacionar la estimación del riesgo, la tarificación y los resultados comerciales mediante visualizaciones interactivas.

**Transición:** Primero explicaremos brevemente los modelos y supuestos que utilizamos.

---

## Integrante 2 — Metodología y modelos
**Tiempo: 0:55–2:00**

**Pantalla:** Metodología y supuestos.

Partimos de un proyecto previo de comparación de modelos de prima pura, utilizando datos sintéticos de seguros de automóviles.

Para desarrollar esta aplicación seleccionamos tres modelos previamente entrenados:

- GLM Tweedie directo, como enfoque estadístico.
- XGBoost directo, como enfoque de Machine Learning.
- MLP Tweedie directo, como enfoque de Deep Learning.

Los tres utilizan seis características del riesgo: edad, zona, cobertura, antigüedad del vehículo, historial de siniestros y tipo de uso.

A partir de sus estimaciones de prima pura construimos las primas comerciales, aplicando la misma política de gastos del 25% y margen objetivo del 5%.

De esta forma comparamos tres alternativas de modelado bajo condiciones comerciales equivalentes.

Además, incorporamos una función probabilística de contratación para estudiar cómo podrían reaccionar los prospectos ante las primas ofrecidas.

Es importante mencionar que la demanda es simulada y que los modelos ya estaban entrenados; nuestro objetivo aquí fue integrar sus resultados en una herramienta de análisis y comunicación.

**Transición:** Ahora veremos cómo estas estimaciones se convierten en cotizaciones individuales.

---

## Integrante 3 — Cotizador individual
**Tiempo: 2:00–3:20**

**Pantalla:** Cotizador individual.

**[Operador: abrir Cotizador individual.]**

En esta sección podemos obtener una cotización a partir de las características de un conductor y su vehículo.

**[Mostrar las seis variables.]**

Primero seleccionamos las características del riesgo que queremos evaluar.

Por ejemplo, utilizaremos este perfil para obtener las primas calculadas por los tres modelos.

**[Operador: pulsar Calcular tres cotizaciones.]**

La aplicación ejecuta los modelos previamente entrenados y genera tres estimaciones de prima pura para el mismo asegurado.

A partir de ellas calcula las correspondientes primas comerciales.

**[Mostrar los resultados.]**

Aquí podemos observar que los precios pueden ser diferentes, aunque las características del asegurado sean exactamente las mismas.

Esto sucede porque cada modelo utiliza un enfoque distinto para estimar el riesgo.

También podemos modificar los supuestos de gastos y margen comercial para observar cómo afectan el precio final.

Finalmente, la aplicación permite exportar las cotizaciones a CSV.

Con ello convertimos un conjunto de modelos predictivos en una herramienta interactiva de tarificación.

**Transición:** Pero una cotización individual no es suficiente para evaluar los resultados de una aseguradora. Por eso desarrollamos también un simulador de cartera.

---

## Integrante 4 — Dashboard de cartera
**Tiempo: 3:20–4:40**

**Pantalla:** Dashboard de cartera.

**[Operador: abrir Dashboard de cartera y seleccionar Valor esperado.]**

Para evaluar las consecuencias comerciales de las primas estimadas, simulamos una cartera de 1,500 prospectos.

Cada prospecto cuenta con características de riesgo, una prima pura estimada por modelo y una probabilidad de aceptar la cotización.

La aplicación permite trabajar con dos perspectivas: contratación simulada, donde se representa una decisión binaria de compra, y contratación esperada, donde ponderamos cada prospecto por su probabilidad de aceptación.

Para esta presentación utilizaremos principalmente la segunda perspectiva.

**[Señalar los indicadores superiores.]**

El dashboard permite comparar indicadores como conversión, prima emitida, siniestros esperados, gastos, resultado técnico y combined ratio.

Este último relaciona los siniestros y gastos con las primas emitidas.

Cuando el combined ratio supera el 100%, significa que, bajo los supuestos del experimento, los costos técnicos exceden los ingresos por primas.

La ventaja de esta herramienta es que podemos comparar los tres modelos utilizando la misma cartera y criterios homogéneos.

**Transición:** Con estos indicadores encontramos diferencias importantes entre las alternativas.

---

## Integrante 5 — Resultados globales y quintiles
**Tiempo: 4:40–5:50**

**Pantalla:** Dashboard de cartera y quintiles de riesgo.

**[Operador: mantener Valor esperado y mostrar la comparación de KPI.]**

En nuestro escenario base encontramos resultados interesantes.

El MLP alcanza la mayor conversión esperada, aproximadamente 58.05%, seguido por GLM con 57.62% y XGBoost con 54.06%.

Sin embargo, al analizar la rentabilidad, el resultado cambia.

El GLM obtiene un resultado técnico positivo de aproximadamente 19,932 unidades monetarias.

En cambio, XGBoost y MLP presentan pérdidas técnicas esperadas cercanas a 74,182 y 125,043 unidades monetarias, respectivamente.

Esto demuestra que **una mayor contratación no necesariamente implica una mayor rentabilidad**.

**[Operador: mostrar el análisis por quintiles.]**

También clasificamos a los prospectos por quintiles de riesgo verdadero, utilizando el oracle de nuestra simulación.

Esto nos permite observar cómo se comportan los modelos entre diferentes niveles de riesgo y no solamente mediante indicadores agregados.

El oracle es una referencia disponible por tratarse de datos sintéticos. En un problema real desconoceríamos la prima pura verdadera individual.

**Transición:** Para profundizar en estas diferencias incorporamos un análisis adicional por segmentos.

---

## Integrante 6 — Análisis por segmentos
**Tiempo: 5:50–8:00**

**Pantalla:** Análisis por segmentos dentro del dashboard.

**[Operador: desplazarse a la sección de segmentación.]**

Una de las ampliaciones más importantes del proyecto fue incorporar un análisis por segmentos.

Consideramos cinco dimensiones: cobertura, zona, historial de siniestros, edad y antigüedad del vehículo.

Esto nos permite estudiar si los modelos presentan diferencias de calibración y rentabilidad entre grupos de asegurados.

**[Operador: mostrar las dimensiones disponibles.]**

Para cada segmento analizamos el sesgo de prima pura estimada respecto al oracle, la conversión, las primas comerciales, la siniestralidad y el resultado técnico esperado.

El sesgo técnico se calcula considerando todos los prospectos de cada grupo, mientras que los indicadores comerciales esperados incorporan las probabilidades de contratación.

Esta distinción es importante porque podemos tener modelos razonablemente calibrados en el conjunto de prospectos, pero presentar resultados desfavorables entre los clientes que terminan contratando.

**[Operador: seleccionar Antigüedad y localizar 0–4 años para MLP.]**

Por ejemplo, para vehículos con antigüedad de cero a cuatro años, el MLP presenta un sesgo técnico de aproximadamente −16.82%.

Esto significa que, en promedio, está subestimando el riesgo verdadero de ese segmento.

Sin embargo, también observamos una conversión esperada de 68.09% y un combined ratio de 113.08%.

Es decir, consigue una contratación elevada, pero las primas resultan insuficientes para cubrir los costos técnicos esperados.

Además, cuando calculamos el sesgo entre los contratantes esperados, la subestimación aumenta hasta aproximadamente −20.52%.

Esto nos permite observar cómo la composición de los clientes que aceptan la cotización puede concentrar riesgos subestimados.

Por lo tanto, el análisis por segmentos complementa la comparación global y nos ayuda a identificar insuficiencias técnicas que podrían quedar ocultas en los promedios de cartera.

**Transición:** Finalmente, evaluamos qué tan estables son nuestras conclusiones cuando modificamos los supuestos de demanda.

---

## Integrante 7 — Sensibilidad de demanda y conclusiones
**Tiempo: 8:00–9:40**

**Pantalla:** Sensibilidad de demanda.

**[Operador: abrir Sensibilidad de demanda. Mostrar inicialmente L = 1.5 y k = 5.]**

Finalmente, quisimos evaluar qué tan sensibles son nuestros resultados a los supuestos utilizados para simular la contratación.

Para ello utilizamos dos parámetros.

El primero es **L**, que controla el nivel hipotético de disposición a pagar de los prospectos.

El segundo es **k**, que determina qué tan pronunciada es la respuesta de contratación frente a la relación entre el precio ofrecido y la disposición a pagar.

**[Señalar los controles.]**

Evaluamos tres valores de cada parámetro, generando nueve escenarios de demanda para cada modelo.

**[Mostrar los mapas de calor.]**

Encontramos que el GLM obtiene el mayor resultado técnico esperado en los nueve escenarios y mantiene resultados positivos en todos ellos.

Sin embargo, esto no significa que el GLM sea universalmente superior, porque nuestras conclusiones dependen del proceso generador de datos, de la cartera sintética y de los supuestos de contratación.

**[Mostrar la tabla de estabilidad.]**

Una pregunta interesante es si este análisis podría utilizarse en una aseguradora real.

La respuesta es sí. Podríamos utilizar una tarifa actuarial vigente como referencia y evaluar modelos alternativos con experiencia histórica de pólizas y siniestros.

Además, reemplazaríamos nuestra función hipotética de demanda por modelos de conversión para nuevos prospectos y modelos de churn o retención para renovaciones, considerando cómo responde la contratación a cambios de precio.

Esto requeriría datos reales y una adecuada calibración del comportamiento comercial.

En conclusión, nuestro proyecto muestra que no basta con evaluar la precisión predictiva. También debemos considerar la suficiencia de primas, la contratación y la composición de la cartera.

Con Streamlit logramos integrar estos elementos en una herramienta interactiva que facilita la comunicación e interpretación de resultados actuariales.

**Muchas gracias por su atención.**

---

## Cierre de la demostración

**[Operador: mantener visible la tabla de estabilidad o regresar al resumen principal.]**

**[No realizar nuevas interacciones mientras se reciben preguntas.]**