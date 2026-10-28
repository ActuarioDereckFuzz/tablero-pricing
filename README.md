# PurePremium Lab

**Tarificación actuarial interactiva de seguros de automóviles con modelos estadísticos, Machine Learning y Deep Learning.**

PurePremium Lab es una aplicación educativa desarrollada con **Streamlit** para explicar cómo las estimaciones de prima pura se traducen en precios comerciales y resultados de cartera. Reutiliza modelos entrenados y evaluados en las fases **01–06** del proyecto *Comparación de modelos para prima pura*; la **fase 07** añade una simulación de contratación, análisis de sensibilidad y evaluación por segmentos, sin reentrenar los estimadores.

**[Abrir aplicación en Streamlit Community Cloud](https://tablero-pricing-kyuoxhtchxnjwsjsygmaj6.streamlit.app/)**

> **Alcance:** demostración académica con datos sintéticos. Todas las cantidades monetarias se expresan en **unidades monetarias simuladas (u.m.)**; no constituyen cotizaciones reales ni resultados de una aseguradora.

## 1. Modelos y diseño del experimento

Se comparan tres estimadores de **prima pura directa**, seleccionados en la fase 06 antes de evaluar su rentabilidad comercial:

| Enfoque | Modelo | Salida utilizada |
|---|---|---|
| Estadístico | GLM Tweedie directo | `pp_tweedie` |
| Machine Learning | XGBoost directo | `pp_xgb_direct` |
| Deep Learning | MLP Tweedie directo | `pp_mlp_tweedie` |

Cada modelo recibe las mismas **seis variables observables**: edad, zona, antigüedad del vehículo, cobertura, historial de siniestros y tipo de uso. Se cotizan los mismos perfiles bajo **una única política comercial**; las diferencias de precio provienen de la prima pura estimada. Por tanto, las tres alternativas **no son políticas fijas denominadas conservadora, competitiva y agresiva**, ni tienen un orden de precio constante entre prospectos.

El *oracle* es la prima pura verdadera conocida únicamente por el proceso generador sintético (*DGP*). **Nunca se utiliza como entrada del cotizador**: sirve para evaluar calibración y costo esperado de siniestros, y participa en la construcción de la demanda hipotética del experimento.

### De la prima pura al precio

La tarifa comercial de cada modelo se calcula como:

$$
P_i^{(m)} = \frac{\widehat{PP}_i^{(m)}}{1-g-r},
$$

con **gastos variables** $g=25\%$ y **margen técnico objetivo** $r=5\%$ en el escenario base. El denominador es $0.70$. El margen del 5% es **un objetivo de tarificación**, no una garantía de rentabilidad observada o esperada, porque el riesgo verdadero puede diferir de la estimación.

## 2. Simulación de contratación y KPI

Se trabaja con una **cartera sintética común de 1,500 prospectos**, pólizas anuales de exposición unitaria. La disposición a pagar es hipotética:

$$
W_i(L)=PP_i^*\,L\exp(0.32Z_i),\qquad
q_i^{(m)}=\frac{1}{1+\exp\!\left[k\log\!\left(P_i^{(m)}/W_i(L)\right)\right]}.
$$

Aquí $PP_i^*$ es el costo esperado *oracle*; $L$ determina el nivel supuesto de disposición a pagar y $k$ la sensibilidad al precio. El escenario base utiliza **$L=1.5$ y $k=5$**. Las alternativas comparten la misma disposición a pagar y el mismo número aleatorio de aceptación por prospecto, para aislar el efecto del precio entre modelos.

La aplicación distingue dos formas de evaluar la contratación:

- **Aceptaciones simuladas:** una compra binaria para cada prospecto según un uniforme común y su probabilidad $q_i^{(m)}$.
- **Valor esperado:** se pondera cada prospecto por $q_i^{(m)}$, sin introducir una realización binaria adicional.

Para ponderaciones $w_i$ binarias o probabilísticas, los KPI principales son:

| KPI | Definición / interpretación |
|---|---|
| Conversión | $\sum_i w_i / 1{,}500$; proporción contratada simulada o esperada |
| Prima emitida | $\sum_i w_i P_i$ |
| Siniestros esperados | $\sum_i w_i PP_i^*$; utiliza el riesgo verdadero sintético |
| Gastos esperados | $g\,\sum_i w_iP_i$ |
| Resultado técnico | Prima emitida − gastos esperados − siniestros esperados |
| *Loss ratio* | Siniestros esperados / prima emitida |
| *Combined ratio* | (Siniestros esperados + gastos esperados) / prima emitida |
| Margen técnico | 1 − *combined ratio* |

Un *combined ratio* superior a **100%** implica pérdida técnica esperada bajo estos supuestos. Las métricas **no incluyen** impuestos, gastos fijos, reaseguro ni costos de capital, y no representan utilidad contable real.

## 3. Aplicación Streamlit

| Sección | Qué permite explorar |
|---|---|
| **Cotizador individual** | Ingresar las seis características, ejecutar los tres modelos persistidos, ajustar gastos/margen del ejemplo y descargar cotizaciones |
| **Dashboard de cartera** | Comparar KPI de 1,500 prospectos, alternar aceptación simulada o valor esperado y explorar conversión por quintiles de riesgo *oracle* |
| **Análisis por segmentos** | Comparar sesgo técnico, MAE, conversión, precio ofertado, siniestralidad, resultado técnico y *combined ratio* por grupos observables; exportar resultados |
| **Sensibilidad de demanda** | Explorar nueve combinaciones de $L\in\{1.3,1.5,1.7\}$ y $k\in\{2.5,5,10\}$, con mapas de calor y tabla de estabilidad |
| **Metodología y límites** | Consultar fórmulas, supuestos, interpretación y restricciones del experimento |

El cotizador **realiza inferencia** con los artefactos guardados. Los indicadores de cartera, sensibilidad y segmentos utilizan los **CSV congelados de 07.01**; cambiar los controles de gastos/margen en el cotizador **no actualiza** esos resultados. El análisis por segmentos permanece fijado al escenario base **$L=1.5$, $k=5$**, incluso al alternar el modo de contratación del dashboard o cambiar parámetros en Sensibilidad.

## 4. Resultados de la corrida base

**Contratación esperada (ponderada por probabilidades), $L=1.5$, $k=5$:**

| Modelo | Conversión | Prima emitida (u.m.) | Resultado técnico (u.m.) | *Combined ratio* |
|---|---:|---:|---:|---:|
| GLM Tweedie | 57.62% | 2,310,217 | **+19,932** | **99.14%** |
| XGBoost directo | 54.06% | 2,279,194 | −74,182 | 103.25% |
| MLP Tweedie | **58.05%** | 2,281,586 | −125,043 | 105.48% |

En esta cartera, **el MLP alcanza la mayor conversión esperada, pero el GLM obtiene el mejor resultado técnico**. Con compras binarias simuladas los resultados técnicos son, respectivamente, **+15,709, −80,989 y −131,841 u.m.**; esas realizaciones no deben confundirse con los valores ponderados de la tabla.

En los **nueve escenarios de sensibilidad**, el GLM alcanza el mayor resultado técnico esperado y es positivo en todos; la conclusión se limita a esa cuadrícula, a la misma cartera y al mecanismo de demanda supuesto.

**Hallazgos segmentados ilustrativos:**

- **Cobertura RC:** los tres modelos presentan *combined ratio* esperado mayor a 100%.
- **Zona 5:** XGBoost y MLP presentan *combined ratios* esperados de aproximadamente **109.4%** y **110.6%**, con subestimación del riesgo entre contratantes esperados.
- **Vehículos de 0–4 años (349 prospectos):** MLP registra sesgo relativo frente al *oracle* de **−16.82%** sobre todos los prospectos y **−20.52%** entre contratantes esperados; su conversión esperada es **68.09%** y su *combined ratio* esperado **113.08%**.
- De los **20 segmentos marginales por modelo**, tienen resultado técnico esperado negativo **8 (GLM), 15 (XGBoost) y 16 (MLP)**. Estas cifras no son porcentajes de clientes afectados ni segmentos independientes.

**Interpretación:** la precisión de prima pura, el precio, la selección de riesgos y la rentabilidad son dimensiones relacionadas, pero distintas. Los resultados **no demuestran superioridad universal** de un modelo ni deben extrapolarse al mercado asegurador real.

## 5. Estructura del repositorio

```text
.
├── app.py                      # Interfaz Streamlit
├── app_core.py                 # Validación y motor de cotización
├── src/models/                 # Inferencia estadística, ML y DL
├── models/                     # Modelos y preprocesadores persistidos
├── results/pricing_simulation/ # Ocho CSV congelados de 07.01
├── tests/test_app_core.py      # Pruebas de consistencia (11 casos)
├── scripts/smoke_inference.py  # Comprobación de 5 perfiles × 3 modelos
├── .streamlit/config.toml      # Apariencia de la aplicación
├── .gitattributes              # Preserva bytes de JSON de artefactos congelados
└── requirements.txt            # Dependencias de despliegue
```

Los **ocho CSV** incluyen seis resultados originales (prospectos, KPI simulados y esperados, quintiles, sensibilidad y ganadores) y dos de la ampliación (`07_01_segment_metrics.csv` y `07_01_segment_findings.csv`). Este repositorio de despliegue **no incluye los notebooks 01–07 ni los datos crudos necesarios para reentrenar**.

## 6. Ejecución local y validación

Se requiere **Python 3.13**, con las dependencias de `requirements.txt`. Desde **Anaconda Prompt en Windows**, en la raíz del repositorio:

```bat
conda create -n purepremium_cloud python=3.13 pip -y
conda activate purepremium_cloud
cd /d C:\ruta\al\repositorio\tablero-pricing

set PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt
python -m pip check
python -m pip install "pytest>=8,<10"
python -m pytest tests/test_app_core.py -q
python scripts/smoke_inference.py
python -m streamlit run app.py
```

Las **11 pruebas unitarias** verifican CSV, identidad comercial, perfiles, segmentos y conciliaciones. El *smoke test* comprueba cinco perfiles con los tres modelos persistidos (**15 predicciones**) sin utilizar el *oracle* como entrada. `PYTHONNOUSERSITE=1` evita depender de paquetes del directorio personal del usuario; es opcional, pero útil para validar el aislamiento del entorno.

En PowerShell, asegúrate de que Conda esté correctamente inicializado o utiliza **Anaconda Prompt**; `conda activate` no es suficiente si la política de ejecución impide cargar los scripts de Conda.

### Publicación

El repositorio se despliega en **[Streamlit Community Cloud](https://share.streamlit.io/)** desde la rama `main`, utilizando `app.py` y **Python 3.13**. El servicio instala las dependencias declaradas en `requirements.txt`; no hay entrenamiento en la nube. Un cambio en GitHub puede requerir un nuevo despliegue o reinicio de la instancia para reflejar el código activo.

## 7. Limitaciones y reproducibilidad

- **DGP y demanda sintéticos:** la disposición a pagar depende del riesgo *oracle*, lo que influye en la selección de riesgos; no se ha calibrado una función de demanda con ventas reales.
- **Evaluación condicionada:** se utiliza una cartera sintética de 1,500 prospectos y nueve supuestos de demanda, sin intervalos de incertidumbre ni validación externa de resultados comerciales.
- **Seguridad e integridad:** los artefactos originales se conservan con controles de hashes SHA-256. Los archivos `.joblib` solo deben cargarse desde fuentes confiables; no se admite carga de modelos arbitrarios por el usuario.
- **Rentabilidad simplificada:** se incluyen siniestros esperados y gastos variables, pero no todos los componentes financieros, operativos o regulatorios de una aseguradora.
- **Reproducibilidad:** las versiones críticas del entorno están fijadas en `requirements.txt`; los rangos de bibliotecas auxiliares no representan un *lockfile* transitivo completo. Ninguna operación de la aplicación reentrena o modifica los modelos.

**Propósito de la entrega:** comunicar resultados actuariales y comerciales de manera trazable mediante visualizaciones e interacciones, sin sustituir la validación técnica de modelos por indicadores exclusivamente comerciales.
