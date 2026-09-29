import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Módulo 5 — Datos: preparación y estructura",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# Estilo visual (CSS + matplotlib)
# ============================================================

COLORES = {
    "azul": "#2E6F9E",
    "naranja": "#E07A5F",
    "verde": "#3D9970",
    "arena": "#F2CC8F",
    "gris": "#667085",
    "oscuro": "#1F2937",
}

st.markdown("""
<style>
.block-container {padding-top: 2rem;}

.hero {
    background: linear-gradient(120deg, #1F4E79 0%, #2E6F9E 55%, #3D9970 100%);
    padding: 1.3rem 1.7rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(31, 78, 121, 0.25);
}
.hero .titulo {color: #FFFFFF; font-size: 1.8rem; font-weight: 700; line-height: 1.2;}
.hero .sub {color: rgba(255, 255, 255, 0.9); font-size: 0.97rem; margin-top: 0.35rem;}

div[data-testid="stMetric"] {
    background: rgba(46, 111, 158, 0.06);
    border: 1px solid rgba(46, 111, 158, 0.20);
    border-left: 5px solid #2E6F9E;
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
}
div[data-testid="stMetricLabel"] p {font-weight: 600;}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(46, 111, 158, 0.10) 0%, rgba(61, 153, 112, 0.05) 100%);
}

.chip {
    display: inline-block;
    padding: 0.18rem 0.7rem;
    margin: 0 0.35rem 0.4rem 0;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.chip-azul    {background: rgba(46, 111, 158, 0.14); color: #2E6F9E;}
.chip-naranja {background: rgba(224, 122, 95, 0.16); color: #C0583D;}
.chip-verde   {background: rgba(61, 153, 112, 0.16); color: #2E7D5B;}
.chip-arena   {background: rgba(242, 204, 143, 0.40); color: #8A6414;}
.chip-gris    {background: rgba(102, 112, 133, 0.14); color: #475467;}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "axes.edgecolor": "#D0D7DE",
    "axes.grid": True,
    "grid.color": "#E6EAEE",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelcolor": "#344054",
    "xtick.color": "#475467",
    "ytick.color": "#475467",
    "font.size": 9,
})


def encabezado(titulo, subtitulo=""):
    """Banner superior de cada página."""
    st.markdown(
        f"<div class='hero'><div class='titulo'>{titulo}</div>"
        f"<div class='sub'>{subtitulo}</div></div>",
        unsafe_allow_html=True,
    )


def mostrar_figura(fig):
    """Ajusta, muestra y libera la figura."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Generación del dataset (sensores IoT sintéticos)
# ============================================================

FEATURES = ["temperatura", "humedad", "presion", "lecturas_hora"]


def generar_dataset(n, semilla, pct_na_temp, pct_na_hum, n_outliers):
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2026-01-01", periods=n, freq="30min")

    tipos_sensor = rng.choice(["DHT22", "BMP180", "LDR"], size=n, p=[0.5, 0.3, 0.2])
    ubicaciones = rng.choice(["invernadero_1", "patio", "bodega"], size=n)
    calidad_senal = rng.choice(["baja", "media", "alta"], size=n, p=[0.1, 0.3, 0.6])

    temperatura = rng.normal(loc=24, scale=3, size=n)
    humedad = rng.normal(loc=60, scale=10, size=n)
    presion = rng.normal(loc=1013, scale=5, size=n)
    lecturas_hora = rng.poisson(lam=12, size=n)

    df = pd.DataFrame({
        "timestamp": fechas,
        "tipo_sensor": tipos_sensor,
        "ubicacion": ubicaciones,
        "calidad_senal": calidad_senal,
        "temperatura": temperatura,
        "humedad": humedad,
        "presion": presion,
        "lecturas_hora": lecturas_hora,
    })

    # Missing values
    if pct_na_temp > 0:
        idx = rng.choice(df.index, size=int(pct_na_temp / 100 * n), replace=False)
        df.loc[idx, "temperatura"] = np.nan
    if pct_na_hum > 0:
        idx = rng.choice(df.index, size=int(pct_na_hum / 100 * n), replace=False)
        df.loc[idx, "humedad"] = np.nan

    # Outliers inyectados en temperatura
    if n_outliers > 0:
        idx = rng.choice(df.index, size=min(n_outliers, n), replace=False)
        df.loc[idx, "temperatura"] = rng.choice([-40, 95, 120], size=len(idx))

    return df


def get_df():
    """Dataset base compartido entre todas las páginas (vía session_state)."""
    cfg = (
        st.session_state.get("n", 500),
        st.session_state.get("semilla", 42),
        st.session_state.get("pct_na_temp", 5),
        st.session_state.get("pct_na_hum", 4),
        st.session_state.get("n_outliers", 6),
    )
    if st.session_state.get("_cfg") != cfg or "df_base" not in st.session_state:
        st.session_state["df_base"] = generar_dataset(*cfg)
        st.session_state["_cfg"] = cfg
    return st.session_state["df_base"]


# ============================================================
# Sidebar — configuración global del dataset
# ============================================================

st.sidebar.title("⚙️ Configuración")
st.sidebar.caption("Sensores IoT sintéticos · temperatura, humedad, presión y lecturas/hora")

st.sidebar.markdown("##### 📐 Tamaño y reproducibilidad")
st.session_state["n"] = st.sidebar.slider("Número de muestras", 100, 2000, 500, step=50)
st.session_state["semilla"] = st.sidebar.number_input("Semilla aleatoria", value=42, step=1)

st.sidebar.markdown("##### 🧪 Imperfecciones del dataset")
st.session_state["pct_na_temp"] = st.sidebar.slider("% missing en temperatura", 0, 30, 5)
st.session_state["pct_na_hum"] = st.sidebar.slider("% missing en humedad", 0, 30, 4)
st.session_state["n_outliers"] = st.sidebar.slider("N° de outliers inyectados (temperatura)", 0, 30, 6)

st.sidebar.divider()
pagina = st.sidebar.radio(
    "🧭 Navegar por el módulo",
    [
        "🏠 Inicio",
        "1️⃣ Tipos de datos",
        "2️⃣ Missing values y outliers",
        "3️⃣ Normalización y estandarización",
        "4️⃣ Train / Val / Test split",
        "5️⃣ Probabilidad y estadística",
    ],
)

df = get_df()

# ============================================================
# PÁGINA: INICIO
# ============================================================
if pagina == "🏠 Inicio":
    encabezado(
        "📊 Módulo 5 — Datos: preparación y estructura",
        "Entender, limpiar y estructurar los datos antes de construir cualquier modelo.",
    )

    st.markdown("""
    Antes de construir cualquier modelo o método computacional, es necesario entender,
    limpiar y estructurar los datos disponibles. Esta aplicación acompaña el notebook del módulo
    y permite **experimentar en vivo** con cada concepto usando un dataset sintético de sensores IoT.

    Usa el menú de la izquierda para:
    - Configurar el dataset (tamaño, % de valores faltantes, outliers inyectados)
    - Navegar por las 5 secciones del módulo
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", len(df))
    c2.metric("Columnas", len(df.columns))
    c3.metric("Missing totales", int(df.isna().sum().sum()))
    c4.metric("Outliers inyectados", int(st.session_state["n_outliers"]))

    with st.container(border=True):
        st.markdown("**👀 Vista previa del dataset (primeras 10 filas)**")
        st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# PÁGINA 1: TIPOS DE DATOS
# ============================================================
elif pagina == "1️⃣ Tipos de datos":
    encabezado(
        "1️⃣ Tipos de datos",
        "Identificar el tipo de cada columna determina qué operaciones tienen sentido sobre ella.",
    )
    st.markdown("""
    Cada columna del dataset representa un tipo distinto: numérico continuo, numérico discreto,
    categórico nominal, categórico ordinal o temporal. Identificarlo correctamente determina
    qué operaciones tienen sentido sobre esa variable.
    """)

    st.markdown(
        "<span class='chip chip-gris'>🕒 Temporal: timestamp</span>"
        "<span class='chip chip-azul'>📈 Continuo: temperatura · humedad · presion</span>"
        "<span class='chip chip-verde'>🔢 Discreto: lecturas_hora</span>"
        "<span class='chip chip-naranja'>🏷️ Nominal: tipo_sensor · ubicacion</span>"
        "<span class='chip chip-arena'>📶 Ordinal: calidad_senal</span>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Tipos originales (detectados por pandas)")
            st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
            mem_antes = df.memory_usage(deep=True).sum() / 1024
            st.metric("Memoria total (antes)", f"{mem_antes:.2f} KB")

    with col2:
        with st.container(border=True):
            st.subheader("Conversión explícita de tipos")
            aplicar = st.toggle("Convertir columnas categóricas a tipo `category`", value=False)
            df_conv = df.copy()
            if aplicar:
                df_conv["tipo_sensor"] = df_conv["tipo_sensor"].astype("category")
                df_conv["ubicacion"] = df_conv["ubicacion"].astype("category")
                orden = ["baja", "media", "alta"]
                df_conv["calidad_senal"] = pd.Categorical(df_conv["calidad_senal"], categories=orden, ordered=True)
            st.dataframe(df_conv.dtypes.astype(str).rename("dtype"), use_container_width=True)
            mem_despues = df_conv.memory_usage(deep=True).sum() / 1024
            st.metric("Memoria total (después)", f"{mem_despues:.2f} KB",
                      delta=f"{mem_despues - mem_antes:.2f} KB", delta_color="inverse")

    st.info("💡 `calidad_senal` es una categoría **ordinal** (baja < media < alta): el orden importa, "
            "a diferencia de `ubicacion` o `tipo_sensor`, que son nominales.")

# ============================================================
# PÁGINA 2: MISSING VALUES Y OUTLIERS
# ============================================================
elif pagina == "2️⃣ Missing values y outliers":
    encabezado(
        "2️⃣ Missing values y outliers",
        "Detectar y tratar valores faltantes y atípicos antes de analizar o modelar.",
    )

    tab_na, tab_out = st.tabs(["🕳️ Missing values", "📦 Outliers — método IQR"])

    with tab_na:
        faltantes = df.isna().sum()
        faltantes_pct = (faltantes / len(df) * 100).round(2)
        tabla_na = pd.DataFrame({"faltantes": faltantes, "% del total": faltantes_pct})[faltantes > 0]

        c_tabla, c_graf = st.columns([1, 1.3])
        with c_tabla:
            with st.container(border=True):
                st.markdown("**Conteo de valores faltantes**")
                if tabla_na.empty:
                    st.success("El dataset no tiene valores faltantes con la configuración actual.")
                else:
                    st.dataframe(tabla_na, use_container_width=True)
        with c_graf:
            if not tabla_na.empty:
                fig, ax = plt.subplots(figsize=(6, 2.2))
                ax.barh(tabla_na.index, tabla_na["% del total"], color=COLORES["naranja"], height=0.5)
                for i, v in enumerate(tabla_na["% del total"]):
                    ax.text(v, i, f"  {v:.1f}%", va="center", fontsize=8, color=COLORES["oscuro"])
                ax.set_xlabel("% de filas con NaN")
                ax.set_title("Porcentaje de faltantes por columna")
                ax.grid(False, axis="y")
                mostrar_figura(fig)

        metodo_imputacion = st.selectbox(
            "Método de tratamiento para 'temperatura' y 'humedad'",
            ["Ninguno (dejar NaN)", "Interpolación lineal", "Eliminar filas (dropna)", "Imputar con la media"],
        )

        df_tratado = df.sort_values("timestamp").reset_index(drop=True).copy()
        if metodo_imputacion == "Interpolación lineal":
            df_tratado[["temperatura", "humedad"]] = df_tratado[["temperatura", "humedad"]].interpolate()
        elif metodo_imputacion == "Eliminar filas (dropna)":
            df_tratado = df_tratado.dropna(subset=["temperatura", "humedad"])
        elif metodo_imputacion == "Imputar con la media":
            df_tratado["temperatura"] = df_tratado["temperatura"].fillna(df_tratado["temperatura"].mean())
            df_tratado["humedad"] = df_tratado["humedad"].fillna(df_tratado["humedad"].mean())

        m1, m2 = st.columns(2)
        m1.metric("Missing restantes tras el tratamiento",
                  int(df_tratado[["temperatura", "humedad"]].isna().sum().sum()))
        dif_filas = len(df_tratado) - len(df)
        m2.metric("Filas tras el tratamiento", len(df_tratado),
                  delta=dif_filas if dif_filas != 0 else None)

    with tab_out:
        c_cfg1, c_cfg2 = st.columns(2)
        variable = c_cfg1.selectbox("Variable a analizar", FEATURES, index=0)
        multiplicador = c_cfg2.slider("Multiplicador IQR", 0.5, 4.0, 1.5, step=0.1)
        st.caption(f"Datos usados: resultado del tratamiento «{metodo_imputacion}» de la pestaña anterior.")

        serie = df_tratado[variable].dropna()
        Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - multiplicador * IQR
        lim_sup = Q3 + multiplicador * IQR
        mask = (serie < lim_inf) | (serie > lim_sup)

        c1, c2, c3 = st.columns(3)
        c1.metric("Límite inferior", f"{lim_inf:.2f}")
        c2.metric("Límite superior", f"{lim_sup:.2f}")
        c3.metric("Outliers detectados", int(mask.sum()))

        estilo_box = dict(
            patch_artist=True,
            widths=0.45,
            boxprops=dict(facecolor=COLORES["azul"], alpha=0.35, edgecolor=COLORES["azul"]),
            medianprops=dict(color=COLORES["naranja"], linewidth=2),
            whiskerprops=dict(color=COLORES["gris"]),
            capprops=dict(color=COLORES["gris"]),
            flierprops=dict(marker="o", markerfacecolor=COLORES["naranja"],
                            markeredgecolor="none", markersize=5, alpha=0.8),
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].boxplot(serie, **estilo_box)
        axes[0].axhline(lim_inf, ls=":", lw=1.3, color=COLORES["verde"], label="Límites IQR")
        axes[0].axhline(lim_sup, ls=":", lw=1.3, color=COLORES["verde"])
        axes[0].legend(fontsize=8, loc="upper right", frameon=False)
        axes[0].set_title(f"{variable} — con outliers")
        axes[1].boxplot(serie[~mask], **estilo_box)
        axes[1].set_title(f"{variable} — sin outliers detectados")
        for ax in axes:
            ax.set_xticks([])
        mostrar_figura(fig)

# ============================================================
# PÁGINA 3: NORMALIZACIÓN Y ESTANDARIZACIÓN
# ============================================================
elif pagina == "3️⃣ Normalización y estandarización":
    encabezado(
        "3️⃣ Normalización y estandarización",
        "Cada fila del dataset es un vector; el dataset completo es una matriz.",
    )
    st.markdown("Aquí puedes comparar el efecto de cada transformación sobre una variable.")

    c_sel1, c_sel2 = st.columns([1, 2])
    variable = c_sel1.selectbox("Variable", FEATURES, index=2)
    metodo = c_sel2.radio("Transformación",
                          ["Ninguna", "Normalización (Min-Max)", "Estandarización (Z-score)"],
                          horizontal=True)

    serie = df[variable].dropna().to_numpy()

    if metodo == "Normalización (Min-Max)":
        transformada = (serie - serie.min()) / (serie.max() - serie.min())
    elif metodo == "Estandarización (Z-score)":
        transformada = (serie - serie.mean()) / serie.std()
    else:
        transformada = serie

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mínimo", f"{transformada.min():.3f}")
    c2.metric("Máximo", f"{transformada.max():.3f}")
    c3.metric("Media", f"{transformada.mean():.3f}")
    c4.metric("Desv. estándar", f"{transformada.std():.3f}")

    if metodo == "Ninguna":
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.hist(transformada, bins=30, color=COLORES["azul"], alpha=0.85, edgecolor="white")
        ax.axvline(transformada.mean(), color=COLORES["naranja"], lw=1.8, ls="--", label="Media")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{variable} — sin transformar")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        axes[0].hist(serie, bins=30, color=COLORES["azul"], alpha=0.85, edgecolor="white")
        axes[0].axvline(serie.mean(), color=COLORES["naranja"], lw=1.8, ls="--", label="Media")
        axes[0].set_title(f"{variable} — original")
        axes[1].hist(transformada, bins=30, color=COLORES["verde"], alpha=0.85, edgecolor="white")
        axes[1].axvline(transformada.mean(), color=COLORES["naranja"], lw=1.8, ls="--", label="Media")
        axes[1].set_title(f"{variable} — {metodo}")
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
    mostrar_figura(fig)

    if metodo != "Ninguna":
        st.caption("La forma de la distribución no cambia: solo cambian la escala y la posición en el eje X.")

# ============================================================
# PÁGINA 4: TRAIN / VAL / TEST SPLIT
# ============================================================
elif pagina == "4️⃣ Train / Val / Test split":
    encabezado(
        "4️⃣ Train / Val / Test split",
        "Separar los datos para entrenar, ajustar y evaluar de forma honesta.",
    )

    tipo_split = st.radio("Tipo de partición", ["Aleatoria", "Cronológica"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    pct_train = col1.slider("% Train", 40, 90, 70)
    pct_val = col2.slider("% Validation", 5, 40, 15)
    pct_test = 100 - pct_train - pct_val
    col3.metric("% Test (calculado)", f"{max(pct_test, 0)}%")

    if pct_test < 0:
        st.error("La suma de Train + Validation supera el 100%. Ajusta los sliders.")
    else:
        df_ordenado = df.sort_values("timestamp").reset_index(drop=True)
        n_total = len(df_ordenado)
        n_train = int(n_total * pct_train / 100)
        n_val = int(n_total * pct_val / 100)

        if tipo_split == "Cronológica":
            train = df_ordenado.iloc[:n_train]
            val = df_ordenado.iloc[n_train:n_train + n_val]
            test = df_ordenado.iloc[n_train + n_val:]
        else:
            barajado = df_ordenado.sample(frac=1, random_state=int(st.session_state["semilla"])).reset_index(drop=True)
            train = barajado.iloc[:n_train]
            val = barajado.iloc[n_train:n_train + n_val]
            test = barajado.iloc[n_train + n_val:]

        partes = [train, val, test]
        etiquetas = ["Train", "Val", "Test"]
        colores = [COLORES["azul"], COLORES["naranja"], COLORES["verde"]]

        # Barra de proporciones
        fig, ax = plt.subplots(figsize=(10, 1.4))
        left = 0
        for parte, color, label in zip(partes, colores, etiquetas):
            size = len(parte)
            ax.barh(0, size, left=left, color=color, edgecolor="white", linewidth=2, height=0.6)
            if size > 0:
                ax.text(left + size / 2, 0, f"{label}\n{size}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += size
        ax.set_xlim(0, n_total)
        ax.axis("off")
        mostrar_figura(fig)

        # Cobertura temporal de cada partición
        fig2, ax2 = plt.subplots(figsize=(10, 2.2))
        for i, (parte, color) in enumerate(zip(partes, colores)):
            ax2.scatter(parte["timestamp"], np.full(len(parte), i),
                        marker="|", s=140, color=color, alpha=0.6)
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(etiquetas)
        ax2.invert_yaxis()
        ax2.grid(False, axis="y")
        ax2.set_title("Cobertura temporal de cada partición")
        mostrar_figura(fig2)

        if tipo_split == "Cronológica":
            r1, r2, r3 = st.columns(3)
            for col, parte, label in zip([r1, r2, r3], partes, ["Train", "Validation", "Test"]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**Rango {label}**")
                        st.caption(f"{parte['timestamp'].min()}  →  {parte['timestamp'].max()}")
            st.info(f"📌 Fecha de corte Train → Validation: **{val['timestamp'].min()}**")
        else:
            st.info("🔀 En la partición aleatoria las tres particiones se mezclan a lo largo de todo el período.")

# ============================================================
# PÁGINA 5: PROBABILIDAD Y ESTADÍSTICA
# ============================================================
elif pagina == "5️⃣ Probabilidad y estadística":
    encabezado(
        "5️⃣ Probabilidad y estadística básica",
        "Medidas descriptivas y relaciones entre variables.",
    )

    resumen = df[FEATURES].agg(["mean", "var", "std"]).T
    resumen.columns = ["media", "varianza", "desv_estandar"]
    with st.container(border=True):
        st.subheader("Estadística descriptiva")
        st.dataframe(resumen.round(3), use_container_width=True)

    col_corr, col_disp = st.columns(2)

    with col_corr:
        with st.container(border=True):
            st.subheader("Matriz de correlación")
            corr = df[FEATURES].corr()
            fig, ax = plt.subplots(figsize=(5, 4.3))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(FEATURES)))
            ax.set_xticklabels(FEATURES, rotation=45, ha="right")
            ax.set_yticks(range(len(FEATURES)))
            ax.set_yticklabels(FEATURES)
            ax.grid(False)
            for i in range(len(FEATURES)):
                for j in range(len(FEATURES)):
                    valor = corr.iloc[i, j]
                    ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(valor) > 0.6 else COLORES["oscuro"])
            fig.colorbar(im, fraction=0.046)
            mostrar_figura(fig)

    with col_disp:
        with st.container(border=True):
            st.subheader("Relación entre dos variables")
            s1, s2 = st.columns(2)
            var_x = s1.selectbox("Variable X", FEATURES, index=0)
            var_y = s2.selectbox("Variable Y", FEATURES, index=1)
            cov_xy = df[[var_x, var_y]].cov().iloc[0, 1]
            corr_xy = df[[var_x, var_y]].corr().iloc[0, 1]

            m1, m2 = st.columns(2)
            m1.metric("Covarianza", f"{cov_xy:.2f}")
            m2.metric("Correlación", f"{corr_xy:.2f}")

            fig2, ax2 = plt.subplots(figsize=(5, 3.4))
            ax2.scatter(df[var_x], df[var_y], alpha=0.45, s=15,
                        color=COLORES["azul"], edgecolors="none")
            ax2.set_xlabel(var_x)
            ax2.set_ylabel(var_y)
            ax2.set_title(f"{var_x} vs {var_y}")
            mostrar_figura(fig2)
