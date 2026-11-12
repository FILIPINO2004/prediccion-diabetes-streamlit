"""App de Streamlit – Predicción temprana de diabetes (Pima Indians Diabetes).
Proyecto de aula · Minería de Datos · Universidad de la Costa (CUC)
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from utils import (CEROS_INVALIDOS, DATASET_PATH, FEATURES, METRICS_PATH, MODEL_PATH,
                   SPECS, TEST_RECORDS_PATH, cargar_datos, predecir, validar_registro)

st.set_page_config(page_title="Predicción temprana de diabetes", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------- carga (una sola vez)
@st.cache_resource
def cargar_pipeline():
    return joblib.load(MODEL_PATH)          # pipeline completo: NO se entrena aquí


@st.cache_data
def cargar_metricas():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_dataset():
    return cargar_datos(DATASET_PATH)


@st.cache_data
def cargar_registros_prueba():
    return pd.read_csv(TEST_RECORDS_PATH)


faltan = [p for p in (MODEL_PATH, METRICS_PATH, DATASET_PATH) if not os.path.exists(p)]
if faltan:
    st.error("Faltan archivos del proyecto: " + ", ".join(faltan) +
             ". Ejecuta el notebook (secciones 9-10) o `python entrenar_modelo.py` y súbelos al repositorio.")
    st.stop()

pipeline, M, df = cargar_pipeline(), cargar_metricas(), cargar_dataset()
UMBRAL = M["umbral"]

# ---------------------------------------------------------------- encabezado
st.title("🩺 Predicción temprana de diabetes a partir de variables clínicas")
st.warning("**Prototipo académico.** Esta herramienta es un ejercicio de Minería de Datos con datos históricos "
           "de un dataset público; **no es un dispositivo médico y no sustituye una decisión ni un diagnóstico profesional.**")
if M.get("sklearn_version") and M["sklearn_version"] != sklearn.__version__:
    st.info(f"El modelo se guardó con scikit-learn {M['sklearn_version']} y la app corre con {sklearn.__version__}. "
            "Si ves resultados extraños, iguala la versión en requirements.txt.")

tab_proy, tab_eda, tab_pred, tab_met = st.tabs(
    ["📋 Proyecto", "📊 Exploración (EDA)", "🧮 Predicción", "✅ Métricas y limitaciones"])

# ---------------------------------------------------------------- 1. proyecto
with tab_proy:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Objetivo")
        st.write("Desarrollar un modelo de clasificación supervisada que estime si una mujer presenta diabetes a partir de "
                 "variables clínicas sencillas de un chequeo rutinario (glucosa, IMC, edad, presión arterial, entre otras), "
                 "**priorizando no dejar pasar casos reales** (alto *recall*).")
        st.subheader("Datos")
        st.write(f"**Pima Indians Diabetes Database** (NIDDK / UCI): {M['n_total']} registros de mujeres de al menos 21 años "
                 "de ascendencia Pima (Arizona, EE. UU.), 8 variables clínicas y la variable objetivo `Outcome` "
                 "(1 = diabetes, 0 = no). Fuente: "
                 "[repositorio de J. Brownlee](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv).")
        st.subheader("Modelo")
        st.write(f"**{M['modelo_final']}** dentro de un `Pipeline` (imputación por mediana"
                 f"{' + escalado' if 'logística' in M['modelo_final'] else ''} + clasificador), con umbral de decisión "
                 f"**{UMBRAL:.2f}** elegido con validación cruzada sobre el conjunto de entrenamiento.")
    with c2:
        st.subheader("Integrantes")
        st.markdown("- Jean Pierre Sucob Ponce\n- Oriana Babilonia\n- Zharick Cammaño")
        st.caption("Minería de Datos · Universidad de la Costa (CUC) · 2026")

# ---------------------------------------------------------------- 2. EDA
with tab_eda:
    st.caption("Los ceros fisiológicamente imposibles se tratan como datos faltantes. Las relaciones mostradas son "
               "asociaciones, no causas.")
    e1, e2 = st.columns(2)
    with e1:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        cont = df["Outcome"].value_counts().sort_index()
        barras = ax.bar(["Sin diabetes (0)", "Con diabetes (1)"], cont.values, color=["#4C72B0", "#C44E52"])
        ax.bar_label(barras)
        ax.set_ylabel("Número de mujeres"); ax.set_title("Distribución de la variable objetivo")
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Desbalance moderado: {100 * cont[1] / cont.sum():.1f} % de casos con diabetes. "
                   "Por eso no basta con el accuracy.")
    with e2:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        datos = [df.loc[df.Outcome == k, "Glucose"].dropna() for k in (0, 1)]
        ax.boxplot(datos, patch_artist=True)
        ax.set_xticks([1, 2], ["Sin diabetes", "Con diabetes"])
        ax.set_ylabel("Glucosa (mg/dL)"); ax.set_title("Glucosa según diagnóstico")
        st.pyplot(fig); plt.close(fig)
        st.caption("La glucosa es la variable que más separa a los dos grupos.")
    e3, e4 = st.columns(2)
    with e3:
        d = df.copy()
        d["IMC"] = pd.cut(d["BMI"], [0, 18.5, 25, 30, 100], right=False,
                          labels=["<18.5", "18.5-24.9", "25-29.9", "≥30"])
        tasa = d.groupby("IMC", observed=True)["Outcome"].agg(["mean", "count"])
        tasa = tasa[tasa["count"] >= 10]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        barras = ax.bar(tasa.index.astype(str), tasa["mean"] * 100, color="#55A868")
        ax.bar_label(barras, fmt="%.1f%%"); ax.set_xlabel("Categoría de IMC (kg/m²)")
        ax.set_ylabel("% con diabetes"); ax.set_title("Diabetes según IMC")
        st.pyplot(fig); plt.close(fig)
        st.caption("Categorías con menos de 10 casos no se muestran.")
    with e4:
        corr = df.corr()["Outcome"].drop("Outcome").sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(corr.index, corr.values, color="#8172B3")
        ax.set_xlabel("Correlación de Pearson con Outcome"); ax.set_title("Asociación con el diagnóstico")
        st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------- 3. predicción
with tab_pred:
    st.subheader("Ingresa los datos clínicos")
    st.caption("Cada campo valida el rango permitido. En las variables que en el dataset original tienen ceros inválidos "
               "puedes marcar «No disponible»: el pipeline usará la mediana de entrenamiento.")

    # estado inicial de los widgets
    for col, s in SPECS.items():
        st.session_state.setdefault(f"v_{col}", s["tipo"](s["default"]))
        if col in CEROS_INVALIDOS:
            st.session_state.setdefault(f"na_{col}", False)

    def cargar_conocido():
        sel = st.session_state["sel_conocido"]
        if sel == "— ninguno —":
            return
        fila = prueba.loc[prueba["fila_dataset"] == int(sel.split()[1])].iloc[0]
        for col, s in SPECS.items():
            val = fila[col]
            if pd.isna(val):
                st.session_state[f"na_{col}"] = True
                st.session_state[f"v_{col}"] = s["tipo"](s["default"])
            else:
                st.session_state[f"v_{col}"] = s["tipo"](val)
                if col in CEROS_INVALIDOS:
                    st.session_state[f"na_{col}"] = False

    if os.path.exists(TEST_RECORDS_PATH):
        prueba = cargar_registros_prueba()
        opciones = ["— ninguno —"] + [f"Registro {int(r.fila_dataset)} (real: {'diabetes' if r.Outcome_real else 'sin diabetes'})"
                                      for r in prueba.itertuples()]
        # el selector usa 'Registro N' como clave; se muestra con el valor real entre paréntesis
        st.selectbox("Cargar un registro conocido del conjunto de prueba (para comparar con el notebook)",
                     [o for o in opciones], key="sel_conocido", on_change=cargar_conocido)

    with st.form("formulario"):
        cols = st.columns(2)
        for i, (col, s) in enumerate(SPECS.items()):
            with cols[i % 2]:
                kw = dict(label=f"{s['etiqueta']} ({s['unidad']})", min_value=s["tipo"](s["min"]),
                          max_value=s["tipo"](s["max"]), step=s["tipo"](s["step"]), key=f"v_{col}",
                          help=f"Rango permitido: {s['min']:g} a {s['max']:g} {s['unidad']}")
                if s["tipo"] is float:
                    kw["format"] = "%.2f" if s["step"] < 0.1 else "%.1f"
                st.number_input(**kw)
                if col in CEROS_INVALIDOS:
                    st.checkbox("No disponible", key=f"na_{col}")
        enviado = st.form_submit_button("Calcular predicción", type="primary")

    if enviado:
        registro = {c: (np.nan if (c in CEROS_INVALIDOS and st.session_state[f"na_{c}"]) else st.session_state[f"v_{c}"])
                    for c in FEATURES}
        errores = validar_registro(registro)
        if errores:
            for e in errores:
                st.error(e)
        else:
            clase, proba = predecir(pipeline, registro, UMBRAL)
            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Probabilidad estimada de diabetes", f"{proba * 100:.1f} %")
                st.caption(f"Umbral de decisión: {UMBRAL * 100:.0f} %")
            with r2:
                if clase == 1:
                    st.error("**Resultado: riesgo alto de diabetes** (probabilidad ≥ umbral). "
                             "Se recomendaría una prueba confirmatoria con un profesional de salud.")
                else:
                    st.success("**Resultado: riesgo bajo de diabetes** (probabilidad < umbral). "
                               "Un resultado bajo no descarta la enfermedad.")
                st.progress(min(max(proba, 0.0), 1.0))
            if any(np.isnan(v) for v in registro.values() if isinstance(v, float)):
                st.caption("Hay variables marcadas como no disponibles: se imputaron con la mediana de entrenamiento.")
            sel = st.session_state.get("sel_conocido", "— ninguno —")
            if sel != "— ninguno —" and os.path.exists(TEST_RECORDS_PATH):
                fila = cargar_registros_prueba().loc[lambda x: x.fila_dataset == int(sel.split()[1])].iloc[0]
                ok = abs(fila.Probabilidad_notebook - proba) < 1e-4
                st.info(f"Verificación con el notebook: probabilidad en el notebook = {fila.Probabilidad_notebook * 100:.2f} % → "
                        + ("**coincide** ✅" if ok else "**no coincide** ⚠️ (¿modificaste algún valor del formulario?)"))

# ---------------------------------------------------------------- 4. métricas
with tab_met:
    T, T2, CV = M["test_final"], M["test_corte2"], M["cv_final"]
    st.subheader(f"Desempeño en el conjunto de prueba (n = {M['n_test']})")
    k = st.columns(5)
    for c, (nombre, clave) in zip(k, [("Accuracy", "Accuracy"), ("Precisión", "Precisión"), ("Recall", "Recall"),
                                      ("F1", "F1"), ("ROC-AUC", "ROC-AUC")]):
        c.metric(nombre, f"{T[clave]:.3f}", f"{T[clave] - T2[clave]:+.3f} vs. corte 2")
    st.caption("La variación compara con el modelo del corte 2 (regresión logística balanceada, umbral 0.5).")

    m1, m2 = st.columns(2)
    with m1:
        cm = np.array([[T["VN"], T["FP"]], [T["FN"], T["VP"]]])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=16,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Sin diabetes", "Con diabetes"]); ax.set_yticks([0, 1], ["Sin diabetes", "Con diabetes"])
        ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real"); ax.set_title("Matriz de confusión (prueba)")
        st.pyplot(fig); plt.close(fig)
    with m2:
        st.write("**Antes vs. después (prueba)**")
        tabla = pd.DataFrame(M["comparacion_test"]).T[["Accuracy", "Precisión", "Recall", "F1", "FN", "FP"]]
        st.dataframe(tabla.style.format({c: "{:.3f}" for c in ["Accuracy", "Precisión", "Recall", "F1"]} | {"FN": "{:.0f}", "FP": "{:.0f}"}))
        st.write("**Validación cruzada (5 pliegues, entrenamiento)**")
        st.dataframe(pd.DataFrame(CV, index=["Valor"]).style.format("{:.3f}"))

    R = M.get("revision")
    if R:
        st.subheader("Revisión de calidad (alerta de la guía: métrica < 0,70)")
        principales = {"Accuracy": T["Accuracy"], "Recall": T["Recall"], "F1": T["F1"]}
        bajas = [k for k, v in principales.items() if v < 0.70]
        if bajas:
            st.warning("Métricas por debajo de 0,70: **" + ", ".join(bajas) + "**. Se revisó el pipeline; el 70 % es una "
                       "señal de revisión, no una garantía de calidad.")
        else:
            st.success("Accuracy, recall y F1 superan 0,70 en la prueba. Aun así se revisan recall por clase, falsos negativos y estabilidad.")
        rv = st.columns(4)
        rv[0].metric("Recall con diabetes", f"{R['recall_con_diabetes']:.3f}")
        rv[1].metric("Recall sin diabetes", f"{R['recall_sin_diabetes']:.3f}")
        rv[2].metric("Brecha F1 (entren. − prueba)", f"{R['brecha_f1']:+.3f}",
                     "posible sobreajuste" if R["brecha_f1"] > 0.10 else "sin sobreajuste evidente", delta_color="off")
        rv[3].metric("Desv. del recall entre pliegues", f"{R['cv_recall_desv']:.3f}")
        st.caption(f"Colinealidad: máxima correlación absoluta entre predictoras = {R['corr_max']:.2f} ({R['corr_max_par']}). "
                   "Fuga de información: imputación y escalado se ajustan solo con entrenamiento dentro del Pipeline.")

    st.subheader("Limitaciones")
    st.markdown(f"""
- El modelo deja pasar **{int(T['FN'])} de {int(T['FN'] + T['VP'])} casos reales** en la prueba y genera **{int(T['FP'])} falsos positivos**.
- El conjunto de prueba es pequeño ({M['n_test']} registros): las métricas tienen **incertidumbre alta**.
- Los datos son de **mujeres Pima de 21 años o más**: no se pueden generalizar a otras poblaciones.
- `Insulin` y `SkinThickness` tienen muchos valores faltantes imputados con la mediana, lo que puede distorsionarlas.
- Se trata de **asociaciones**, no de causalidad.
- {"El F1 está por debajo de 0.70: se mantiene como señal de revisión." if T["F1"] < 0.70 else "Todas las métricas principales superan 0.70, pero eso no garantiza calidad clínica."}
- **Prototipo académico**: no sustituye una decisión profesional.
""")
