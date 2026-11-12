# Predicción temprana de diabetes – App en Streamlit

Proyecto de aula · Minería de Datos · Universidad de la Costa (CUC) · Actividad 1 – Tercer corte
**Integrantes:** Jean Pierre Sucob Ponce · Oriana Babilonia · Zharick Cammaño
**App pública:** _(pegar aquí la URL de Streamlit Community Cloud)_

> ⚠️ **Prototipo académico.** No es un dispositivo médico y no sustituye una decisión ni un diagnóstico profesional.

## Descripción
App web que usa el pipeline entrenado en el corte 2 (imputación por mediana + escalado/modelo) para estimar la probabilidad de diabetes a partir de 8 variables clínicas. **La app no entrena: carga el pipeline guardado con `joblib`.**

- **Datos:** Pima Indians Diabetes Database (768 registros, mujeres ≥ 21 años). Archivo: `pima-indians-diabetes.csv` (mismo nombre que usa el notebook).
- **Objetivo:** `Outcome` (1 = diabetes). Se prioriza el *recall* para no dejar pasar casos reales.
- **Modelo y umbral:** ver pestaña «Métricas» de la app (se generan en `models/metricas.json`).

## Estructura
| Archivo | Contenido |
|---|---|
| `app.py` | App de Streamlit (proyecto, EDA, formulario, predicción, métricas, limitaciones) |
| `utils.py` | Rangos, validación y función de predicción compartidos |
| `entrenar_modelo.py` | Reproduce las secciones 9–10 del notebook y guarda el pipeline |
| `Avance3_Prediccion_diabetes_Js.ipynb` | Notebook del corte 2 + revisión/optimización del corte 3 |
| `pima-indians-diabetes.csv` | Dataset (sin encabezado) |
| `models/pipeline_diabetes.joblib` | Pipeline completo serializado |
| `models/metricas.json`, `models/registros_prueba.csv` | Métricas y registros de prueba para verificar app vs. notebook |

## Cómo reproducir
1. Colocar `pima-indians-diabetes.csv` en la raíz.
2. Ejecutar el notebook completo (o `python entrenar_modelo.py`) → genera la carpeta `models/`.
3. `pip install -r requirements.txt` y `streamlit run app.py`.
4. **Importante:** la versión de `scikit-learn` de `requirements.txt` debe ser la misma con la que se guardó el modelo (el notebook la imprime al final).

## Publicación
Subir todo a un repositorio público de GitHub → share.streamlit.io → *New app* → repositorio, rama `main`, archivo `app.py`.

## Pruebas de funcionamiento
- Caso normal: valores por defecto del formulario.
- Entrada inválida: p. ej. Glucosa = 500 (el formulario bloquea el rango y `validar_registro` lo rechaza).
- Registro conocido: en la pestaña «Predicción» elegir un registro del conjunto de prueba; la app muestra si la probabilidad coincide con la del notebook.

## Uso de inteligencia artificial
| Campo | Detalle |
|---|---|
| Herramienta | Claude (Anthropic) |
| Propósito | Apoyo para estructurar `app.py`, `utils.py`, el script de entrenamiento y la sección 9–10 del notebook |
| Prompt principal | _(pegar el prompt real utilizado)_ |
| Cómo se verificó | Se ejecutó el código de entrenamiento de punta a punta, se recargó el pipeline y se comprobó que la probabilidad coincide con la del notebook; se revisaron rangos y validaciones |
| Problemas encontrados | Se adjuntó por error un dataset distinto al del proyecto (sintético); la versión de scikit-learn debe coincidir entre entrenamiento y despliegue |
| Ajustes realizados | Se rehízo todo sobre el dataset Pima y el notebook del corte 2; se agregó revisión por la alerta de 0,70 (F1, recall por clase, sobreajuste, estabilidad, colinealidad) y ajuste de umbral usando solo datos de entrenamiento |
| Comprensión | Todos los integrantes deben poder explicar el código entregado |
