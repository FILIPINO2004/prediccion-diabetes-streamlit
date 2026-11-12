"""Entrena y guarda el pipeline completo. Equivale a la sección 9-10 del notebook
Avance3_Prediccion_diabetes_Js.ipynb (misma partición, semilla y criterios).
Uso:  python entrenar_modelo.py   (requiere pima-indians-diabetes.csv en esta carpeta)
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from utils import COLUMNAS, CEROS_INVALIDOS

SEED = 42
df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=COLUMNAS)
df_na = df.copy()
df_na[CEROS_INVALIDOS] = df_na[CEROS_INVALIDOS].replace(0, np.nan)
X, y = df_na.drop(columns="Outcome"), df_na["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
modelos = {"Regresión logística (balanceada)": Pipeline([
    ("imputador", SimpleImputer(strategy="median")), ("escalador", StandardScaler()),
    ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))])}
modelos["Regresión logística (balanceada)"].fit(X_train, y_train)

import json
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import make_scorer, fbeta_score

DATASET_PATH = "pima-indians-diabetes.csv"      # mismo nombre que usa la app de Streamlit
os.makedirs("models", exist_ok=True)
modelo_corte2 = modelos["Regresión logística (balanceada)"]   # modelo del corte 2 (referencia "antes")

f2 = make_scorer(fbeta_score, beta=2)
scoring = {"f2": f2, "recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}

candidatos = {
    "Regresión logística (balanceada, ajustada)": (
        Pipeline([("imputador", SimpleImputer(strategy="median")), ("escalador", StandardScaler()),
                  ("modelo", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]),
        {"modelo__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}),
    "Bosque aleatorio (balanceado)": (
        Pipeline([("imputador", SimpleImputer(strategy="median")),
                  ("modelo", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=SEED))]),
        {"modelo__max_depth": [3, 4, 6, 8], "modelo__min_samples_leaf": [5, 10, 20]}),
}

busquedas, filas = {}, []
for nombre, (pipe, grid) in candidatos.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, refit="f2", n_jobs=-1)
    gs.fit(X_train, y_train)
    busquedas[nombre] = gs
    i = gs.best_index_
    filas.append({"Modelo": nombre, "Mejores parámetros": gs.best_params_,
                  "F2 (CV)": gs.cv_results_["mean_test_f2"][i],
                  "Recall (CV)": gs.cv_results_["mean_test_recall"][i],
                  "Precisión (CV)": gs.cv_results_["mean_test_precision"][i],
                  "F1 (CV)": gs.cv_results_["mean_test_f1"][i],
                  "ROC-AUC (CV)": gs.cv_results_["mean_test_roc_auc"][i]})
tabla_busqueda = pd.DataFrame(filas).set_index("Modelo")
print("Tabla 8. Mejor configuración de cada modelo (validación cruzada de 5 pliegues, solo entrenamiento)")
print(tabla_busqueda.round(3).to_string())

# Modelo con mayor F2 en validación cruzada
nombre_final = max(busquedas, key=lambda n: busquedas[n].best_score_)
modelo_final = busquedas[nombre_final].best_estimator_
print("Modelo seleccionado:", nombre_final, busquedas[nombre_final].best_params_)

# Umbral: se elige con probabilidades out-of-fold del ENTRENAMIENTO (la prueba no interviene)
proba_oof = cross_val_predict(modelo_final, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
umbrales = np.round(np.arange(0.20, 0.81, 0.01), 2)
filas = []
for t in umbrales:
    p = (proba_oof >= t).astype(int)
    filas.append({"umbral": t, "F2": fbeta_score(y_train, p, beta=2),
                  "recall": recall_score(y_train, p), "precision": precision_score(y_train, p, zero_division=0)})
curva_umbral = pd.DataFrame(filas).set_index("umbral")
UMBRAL = float(curva_umbral["F2"].idxmax())
print(f"Umbral elegido (máximo F2 en entrenamiento, out-of-fold): {UMBRAL}")

ax = curva_umbral.plot(figsize=(7, 3.8))
ax.axvline(UMBRAL, color="gray", linestyle="--")
plt.title("Figura 9. Métricas según el umbral de decisión (validación cruzada en entrenamiento)")
plt.xlabel("Umbral de probabilidad"); plt.tight_layout(); plt.close('all')

def evaluar(y_true, proba, umbral):
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"Accuracy": accuracy_score(y_true, pred), "Precisión": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred), "F1": f1_score(y_true, pred),
            "F2": fbeta_score(y_true, pred, beta=2), "ROC-AUC": roc_auc_score(y_true, proba),
            "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn)}

proba_antes = modelo_corte2.predict_proba(X_test)[:, 1]
proba_final = modelo_final.predict_proba(X_test)[:, 1]
comparacion = pd.DataFrame({
    "Corte 2: reg. logística balanceada (umbral 0.5)": evaluar(y_test, proba_antes, 0.5),
    f"Corte 3: {nombre_final} (umbral 0.5)": evaluar(y_test, proba_final, 0.5),
    f"Corte 3: {nombre_final} (umbral {UMBRAL})": evaluar(y_test, proba_final, UMBRAL),
}).T
print("Tabla 9. Métricas en el conjunto de prueba (clase positiva = diabetes)")
print(comparacion.round(3).to_string())

# Alerta de revisión: ¿alguna métrica principal por debajo de 0.70?
final = comparacion.iloc[-1]
for m in ["Accuracy", "Recall", "F1"]:
    estado = "OK" if final[m] >= 0.70 else "REVISAR (< 0.70)"
    print(f"{m:10s} {final[m]:.3f}  {estado}")
print(f"Falsos negativos: {int(final['FN'])} de {int(final['FN'] + final['VP'])} casos reales")

pred_final = (proba_final >= UMBRAL).astype(int)
ConfusionMatrixDisplay.from_predictions(y_test, pred_final, display_labels=["Sin diab.", "Con diab."],
                                        cmap="Blues", colorbar=False)
plt.title(f"Figura 10. Matriz de confusión final (umbral {UMBRAL}, n = {len(y_test)})")
plt.tight_layout(); plt.close('all')

# Revisión de la Ficha técnica: recall por clase, sobreajuste, estabilidad y colinealidad
rec_pos = recall_score(y_test, pred_final)                  # sensibilidad (con diabetes)
rec_neg = recall_score(y_test, pred_final, pos_label=0)     # especificidad (sin diabetes)

pred_train = (modelo_final.predict_proba(X_train)[:, 1] >= UMBRAL).astype(int)
f1_train, f1_test = f1_score(y_train, pred_train), f1_score(y_test, pred_final)

f1_folds, rec_folds = [], []                                # estabilidad por pliegue (umbral final)
for _, idx in cv.split(X_train, y_train):
    p_ = (proba_oof[idx] >= UMBRAL).astype(int)
    f1_folds.append(f1_score(y_train.iloc[idx], p_)); rec_folds.append(recall_score(y_train.iloc[idx], p_))

c_abs = X.corr().abs().where(~np.eye(X.shape[1], dtype=bool))
par = c_abs.stack().idxmax(); corr_max = float(c_abs.stack().max())

revision = {
    "recall_con_diabetes": round(rec_pos, 4), "recall_sin_diabetes": round(rec_neg, 4),
    "f1_entrenamiento": round(f1_train, 4), "f1_prueba": round(f1_test, 4),
    "brecha_f1": round(f1_train - f1_test, 4),
    "cv_recall_media": round(float(np.mean(rec_folds)), 4), "cv_recall_desv": round(float(np.std(rec_folds)), 4),
    "cv_f1_media": round(float(np.mean(f1_folds)), 4), "cv_f1_desv": round(float(np.std(f1_folds)), 4),
    "corr_max": round(corr_max, 3), "corr_max_par": f"{par[0]} - {par[1]}",
}
for k_, v_ in revision.items(): print(f"{k_:22s} {v_}")
print("\nSobreajuste:", "posible (brecha F1 > 0.10)" if revision["brecha_f1"] > 0.10 else "no evidente (brecha F1 <= 0.10)")
print("Estabilidad: desv. del recall entre pliegues =", revision["cv_recall_desv"])
print("Colinealidad:", "alta (> 0.80)" if corr_max > 0.80 else "no problemática (máx |r| <= 0.80)")

# El pipeline incluye imputación, escalado (si aplica) y modelo: la app NO vuelve a entrenar.
joblib.dump(modelo_final, "models/pipeline_diabetes.joblib")

def a_float(d): return {k: (int(v) if k in ("VP", "FP", "FN", "VN") else round(float(v), 4)) for k, v in d.items()}
cv_final = busquedas[nombre_final]; i = cv_final.best_index_
metricas = {
    "modelo_final": nombre_final,
    "hiperparametros": {k: (v.item() if hasattr(v, "item") else v) for k, v in cv_final.best_params_.items()},
    "umbral": UMBRAL,
    "sklearn_version": sklearn.__version__,
    "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "pct_diabetes": round(float(y.mean() * 100), 1),
    "test_final": a_float(comparacion.iloc[-1].to_dict()),
    "test_corte2": a_float(comparacion.iloc[0].to_dict()),
    "cv_final": {"F2": round(float(cv_final.cv_results_["mean_test_f2"][i]), 4),
                 "Recall": round(float(cv_final.cv_results_["mean_test_recall"][i]), 4),
                 "Precisión": round(float(cv_final.cv_results_["mean_test_precision"][i]), 4),
                 "F1": round(float(cv_final.cv_results_["mean_test_f1"][i]), 4),
                 "ROC-AUC": round(float(cv_final.cv_results_["mean_test_roc_auc"][i]), 4)},
    "revision": revision,
    "comparacion_test": {k: a_float(v) for k, v in comparacion.to_dict("index").items()},
}
with open("models/metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

# Registros de prueba con la predicción del notebook, para verificar la app (misma entrada → misma salida)
prueba = X_test.copy()
prueba["Outcome_real"] = y_test
prueba["Probabilidad_notebook"] = proba_final.round(6)
prueba["Prediccion_notebook"] = pred_final
prueba.index.name = "fila_dataset"
prueba.to_csv("models/registros_prueba.csv")
print("Guardados: models/pipeline_diabetes.joblib, models/metricas.json, models/registros_prueba.csv")
print("Versión de scikit-learn usada:", sklearn.__version__, "(debe coincidir con requirements.txt)")

pipe_cargado = joblib.load("models/pipeline_diabetes.joblib")
registro = X_test.iloc[[0]]                              # un registro del conjunto de prueba
p = pipe_cargado.predict_proba(registro)[0, 1]
print("Probabilidad de diabetes (pipeline recargado):", round(p, 6))
print("Probabilidad en el notebook:                  ", round(proba_final[0], 6))
assert abs(p - proba_final[0]) < 1e-9
print("Predicción:", "diabetes" if p >= UMBRAL else "sin diabetes", f"(umbral {UMBRAL})")
