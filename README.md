# PurePremium Lab — Streamlit (paquete de despliegue)

Aplicación **educativa** de tarificación de seguros de autos. Compara GLM Tweedie,
XGBoost directo y MLP Tweedie, con una misma política comercial (25% de gastos,
5% de margen objetivo). Los resultados de 07.01 provienen de **1,500 prospectos
sintéticos** y nueve escenarios hipotéticos de demanda.

Este paquete se generó a partir de la versión final `app.zip` de la fase 07.02.
**No entrena modelos** ni modifica sus pesos, hashes, cálculos o CSV.
No contiene los notebooks ni los datos crudos necesarios para reentrenar.

## Contenido

- `app.py`: interfaz con cotizador, dashboard, sensibilidad y metodología.
- `app_core.py` y `src/models/`: lógica de tarificación y los tres motores de inferencia originales.
- `models/`: artefactos congelados y manifiestos con controles SHA256.
- `results/pricing_simulation/`: seis CSV de la simulación 07.01.
- `.streamlit/config.toml`: configuración visual actual, **provisional** hasta acuerdo del equipo.
- `requirements.txt`: único manifiesto de dependencias reconocido por Streamlit Community Cloud.
- `tests/test_app_core.py`, `scripts/smoke_inference.py`: pruebas.

## 1. Instalación local limpia (Windows / PowerShell)

Requisitos: Conda o Miniconda. Extrae el ZIP y **abre la terminal dentro de la carpeta
que contiene `app.py`** (no dentro de `models` ni `src`).

```powershell
conda create -n purepremium_cloud python=3.11 pip -y
conda activate purepremium_cloud
python --version
python -m pip install -r requirements.txt
python -m pip check
```

La versión de Python deberá ser **3.11.x**. Los modelos contienen artefactos
serializados con las versiones exactas de NumPy, pandas, sklearn, XGBoost,
joblib, statsmodels y TensorFlow indicadas en `requirements.txt`.

## 2. Comprobaciones de funcionamiento

```powershell
python -m pip install "pytest>=8,<10"
python -m pytest tests/test_app_core.py -q
python scripts/smoke_inference.py
python -m streamlit run app.py
```

- `pytest` comprueba contratos y consistencia de resultados 07.01.
- El *smoke test* carga **los tres modelos reales** y verifica cinco prospectos
  contra sus predicciones guardadas (15 verificaciones). Es obligatorio para
  dar por verificada la inferencia completa, pero no requiere el dataset original.
- En Streamlit, comprueba las cuatro secciones y cotiza un perfil en el formulario.
- No asumimos que una prueba satisfactoria en Python 3.13 o sin TensorFlow equivalga
  a un smoke test en Python 3.11 con todas las dependencias.

Para salir del entorno: `conda deactivate`.

## 3. Publicar en Streamlit Community Cloud

1. Comprueba las pruebas locales antes de publicar.
2. Crea un repositorio de GitHub y sube **el contenido de esta carpeta** (no el archivo ZIP).
   `app.py` y `requirements.txt` deben quedar en la raíz del repositorio.
3. En [share.streamlit.io](https://share.streamlit.io/), crea una app desde ese repositorio.
4. Selecciona la rama y **Main file path: `app.py`**.
5. En **Advanced settings**, selecciona **Python 3.11** antes de desplegar.
6. Despliega y revisa los logs. Verifica la cotización de las tres estrategias
   y el dashboard; el arranque/carga de TensorFlow puede consumir memoria.

Este paquete evita incluir `environment.yml` para que Community Cloud use
`requirements.txt` directamente. No se necesita `packages.txt` mientras las
ruedas Linux de las dependencias se instalen sin errores; si aparecen librerías
sistema faltantes, revisa el log antes de añadir paquetes del sistema.

## 4. Problemas frecuentes

| Síntoma | Revisión recomendada |
|---|---|
| `ModuleNotFoundError` | Confirma que el entorno Conda está activo y ejecutaste `pip install -r requirements.txt`. |
| Incompatibilidad pickle/joblib | No cambies las versiones críticas del archivo de dependencias ni los artefactos. |
| Hash SHA256 diferente | Verifica que GitHub contiene los archivos `.joblib`, `.keras` y manifiestos intactos. |
| La app funciona pero no cotiza | Ejecuta `python scripts/smoke_inference.py`; examina el error al cargar TensorFlow o modelos. |
| Out of memory en Community Cloud | Consulta logs y límites; no cambies silenciosamente los modelos por predicciones precalculadas. |
| Conflicto de dependencias | Revisa `python -m pip check` y la versión efectiva de Python. |

## 5. Límites y reproducibilidad

- `requirements.txt` conserva versiones críticas del entrenamiento; rangos
  para librerías auxiliares **no constituyen un lockfile transitivo**.
- El funcionamiento en Community Cloud depende de disponibilidad de paquetes
  Linux, RAM y recursos de la plataforma; **no se garantiza hasta desplegar**.
- El dashboard lee CSV congelados; el cotizador hace inferencia de verdad.
- La prima comercial individual se calcula como `prima_pura / (1-g-m)`, no es
  una cotización contractual ni una recomendación de compra.
- La demanda, aceptación y riesgo verdadero (*oracle*) son simulados.
- Los archivos `.joblib` usan pickle: carga artefactos solo de un repositorio
  bajo tu control y no habilites subidas de modelos de origen desconocido.
- Este empaquetado **no altera el diseño visual**; se revisará con el equipo.

Fuente del criterio de instalación: documentación oficial de Streamlit Community Cloud,
[dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
y [publicación](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
