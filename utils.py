"""Constantes y funciones compartidas por la app de Streamlit y el entrenamiento.

No importa streamlit, así que se puede probar desde el notebook o la terminal.
"""
import numpy as np
import pandas as pd

# Mismo nombre de archivo que usa el notebook (celda de carga de datos).
DATASET_PATH = "pima-indians-diabetes.csv"
COLUMNAS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURES = COLUMNAS[:-1]
TARGET = "Outcome"
# Variables donde un 0 es imposible fisiológicamente (faltante disfrazado)
CEROS_INVALIDOS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

MODEL_PATH = "models/pipeline_diabetes.joblib"
METRICS_PATH = "models/metricas.json"
TEST_RECORDS_PATH = "models/registros_prueba.csv"

# tipo, mínimo, máximo, valor por defecto (≈ mediana), paso, etiqueta y unidad
SPECS = {
    "Pregnancies":              dict(tipo=int,   min=0,    max=20,    default=3,     step=1,    etiqueta="Número de embarazos", unidad="embarazos"),
    "Glucose":                  dict(tipo=float, min=40.0, max=300.0, default=117.0, step=1.0,  etiqueta="Glucosa (prueba de tolerancia)", unidad="mg/dL"),
    "BloodPressure":            dict(tipo=float, min=30.0, max=140.0, default=72.0,  step=1.0,  etiqueta="Presión arterial diastólica", unidad="mm Hg"),
    "SkinThickness":            dict(tipo=float, min=5.0,  max=100.0, default=29.0,  step=1.0,  etiqueta="Pliegue cutáneo del tríceps", unidad="mm"),
    "Insulin":                  dict(tipo=float, min=10.0, max=900.0, default=125.0, step=5.0,  etiqueta="Insulina sérica (2 h)", unidad="µU/mL"),
    "BMI":                      dict(tipo=float, min=10.0, max=70.0,  default=32.0,  step=0.1,  etiqueta="Índice de masa corporal (IMC)", unidad="kg/m²"),
    "DiabetesPedigreeFunction": dict(tipo=float, min=0.05, max=2.5,   default=0.37,  step=0.01, etiqueta="Función de pedigrí de diabetes", unidad="índice"),
    "Age":                      dict(tipo=int,   min=21,   max=90,    default=29,    step=1,    etiqueta="Edad", unidad="años"),
}


def cargar_datos(path: str = DATASET_PATH) -> pd.DataFrame:
    """CSV sin encabezado; los ceros inválidos se reemplazan por NaN (como en el notebook)."""
    df = pd.read_csv(path, header=None, names=COLUMNAS)
    df[CEROS_INVALIDOS] = df[CEROS_INVALIDOS].replace(0, np.nan)
    return df


def validar_registro(registro: dict) -> list:
    """Devuelve la lista de errores (vacía si el registro es válido).
    Las cinco variables con 'ceros inválidos' pueden venir como NaN (dato no disponible):
    el pipeline las imputa con la mediana de entrenamiento."""
    errores = []
    for col, s in SPECS.items():
        v = registro.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if col in CEROS_INVALIDOS:
                continue
            errores.append(f"Falta el valor de «{s['etiqueta']}».")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errores.append(f"«{s['etiqueta']}» debe ser numérico.")
            continue
        if not s["min"] <= v <= s["max"]:
            errores.append(f"«{s['etiqueta']}» = {v:g} está fuera de rango "
                           f"({s['min']:g} a {s['max']:g} {s['unidad']}).")
        elif s["tipo"] is int and v != int(v):
            errores.append(f"«{s['etiqueta']}» debe ser un número entero.")
    return errores


def predecir(pipeline, registro: dict, umbral: float = 0.5):
    """Aplica el pipeline COMPLETO (imputación + escalado + modelo) a un registro.
    Devuelve (clase 0/1, probabilidad de diabetes)."""
    X = pd.DataFrame([registro], columns=FEATURES).astype(float)
    proba = float(pipeline.predict_proba(X)[0, 1])
    return int(proba >= umbral), proba
