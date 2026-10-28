"""07.02 — Aplicación de tarificación educativa de tres modelos congelados.

Ejecutar desde esta carpeta: streamlit run app.py
- Dashboard: utiliza los seis CSV originales y dos CSV de segmentación 07.01.
- Cotizador: ejecuta los modelos reales, sin oracle ni reentrenamiento.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_core import (
    DATASETS, DEFAULT_EXPENSE_RATIO, DEFAULT_TARGET_MARGIN, MODEL_COLUMNS,
    SEGMENT_ORDER, STRATEGIES, PricingEngine, load_simulation,
)

ROOT = Path(__file__).resolve().parent
NAMES = {
    "Estadístico · GLM Tweedie": "GLM Tweedie",
    "ML · XGBoost directo": "XGBoost",
    "DL · MLP Tweedie": "MLP Tweedie",
}
COLORS = ["#5CE1C4", "#86B0FF", "#B9A1FF"]

st.set_page_config(
    page_title="PurePremium Lab | Tarificación",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', Inter, sans-serif; }
.block-container { max-width: 1290px; padding-top: 1.9rem; padding-bottom: 3.4rem; }
h1,h2,h3 { letter-spacing: -0.035em; }
header[data-testid="stHeader"] { background: rgba(10,16,32,.75); }
.stApp { background: radial-gradient(ellipse at 84% 2%, rgba(35,79,105,.24), transparent 38%),
                     radial-gradient(ellipse at 16% 20%, rgba(26,76,67,.17), transparent 32%), #0A1020; }
.hero { background: linear-gradient(114deg, rgba(34,69,83,.58), rgba(17,30,58,.82));
    border: 1px solid rgba(133,211,220,.17); border-radius: 20px;
    padding: 1.6rem 1.75rem; margin-bottom: 1.35rem; }
.hero .eyebrow { color:#5CE1C4; font-size:.74rem; font-weight:800;
    letter-spacing:.19em; text-transform:uppercase; margin-bottom:.55rem; }
.hero h1 { color: #F1F7FF; font-size:2.2rem; line-height:1.13; margin:0 0 .4rem; }
.hero p { color:#B7C9DB; margin:0; max-width:68rem; }
.section-note { color:#B2C5D7; font-size:.92rem; margin-bottom:1rem; }
[data-testid="stMetric"] { background:rgba(19,32,51,.72);
   border:1px solid rgba(154,184,220,.12); padding:1.05rem 1rem; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#A9BED3; }
[data-testid="stMetricValue"] { font-size:1.56rem; }
[data-testid="stSidebar"] { background: #0F192B; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
   color:#05271F; font-weight:750; }
div[data-testid="stAlert"] { border-radius: 12px; }
.footnote { border-top:1px solid #27394B; margin-top:2.2rem; padding-top:.75rem;
    color:#93A9C3; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def data_tables() -> dict[str, pd.DataFrame]:
    return load_simulation(ROOT)


@st.cache_resource(show_spinner="Cargando modelos")
def engine() -> PricingEngine:
    return PricingEngine.from_root(ROOT)


def um(amount: float) -> str:
    return f"{amount:,.0f} u.m."


def pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}%}"


def plot_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(strokeOpacity=0).configure_axis(
        labelColor="#B6C8DA", titleColor="#B6C8DA", gridColor="#26384E"
    ).configure_legend(labelColor="#D7E5F2", titleColor="#D7E5F2")


def add_short_name(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["modelo"] = result["estrategia"].map(NAMES)
    return result


try:
    tables = data_tables()
except Exception as exc:
    st.error(f"No fue posible cargar/validar los CSV: {type(exc).__name__}: {exc}")
    st.stop()

n_prospects = len(tables["prospects"])
with st.sidebar:
    st.markdown("### PUREPREMIUM LAB")
    st.caption("DEMO ACTUARIAL")
    st.divider()
    page = st.radio(
        "Explorar aplicación",
        ["Cotizador individual", "Dashboard de cartera", "Sensibilidad de demanda", "Metodología"],
        index=0,
    )
    st.divider()
    st.caption("CORRIDA CONGELADA")
    st.markdown(f"**{n_prospects:,}** prospectos sintéticos")
    st.caption("semilla 20261022")
    st.caption("3 modelos · 9 escenarios de demanda")
    st.caption("Sin reentrenamiento de modelos")
    st.divider()
    st.caption("Resultados explicativos, no un producto de seguros ni precios de mercado.")

st.markdown("""
<div class="hero">
 <div class="eyebrow">Actuarial analytics &nbsp; / &nbsp; pricing intelligence</div>
 <h1>El riesgo se estima. El precio se decide.</h1>
 <p>Una misma cartera, tres modelos de prima pura y una política comercial común.
    Compara cotizaciones, contratación y resultado técnico sin confundir precisión predictiva con rentabilidad.</p>
</div>
""", unsafe_allow_html=True)

if page == "Cotizador individual":
    st.subheader("Cotizador individual")
    st.markdown('<p class="section-note">Introduce seis características observables. Se cargan exclusivamente los modelos previamente entrenados y guardados; las variables <i>oracle</i> no se utilizan para cotizar.</p>', unsafe_allow_html=True)

    zone_labels = {1: "Muy bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy alto"}
    coverage_labels = {0: "Limitada", 1: "RC", 2: "Amplia"}
    usage_labels = {0: "Particular", 1: "Transporte privado", 2: "Carga comercial"}
    with st.form("individual_quote", clear_on_submit=False):
        a, b, c = st.columns(3)
        with a:
            edad = st.number_input("Edad del conductor (años)", min_value=18, max_value=74, value=40, step=1)
            zona = st.selectbox("Zona de riesgo", options=list(zone_labels), index=2,
                                format_func=lambda key: f"{key} · {zone_labels[key]}")
        with b:
            antiguedad = st.number_input("Antigüedad del vehículo (años)", min_value=0, max_value=19, value=5, step=1)
            cobertura = st.selectbox("Cobertura", options=list(coverage_labels), index=2,
                                    format_func=lambda key: coverage_labels[key])
        with c:
            historial = st.number_input("Siniestros previos", min_value=0, max_value=4, value=0, step=1)
            tipo_uso = st.selectbox("Tipo de uso", options=list(usage_labels), index=0,
                                   format_func=lambda key: usage_labels[key])
        with st.expander("Política comercial: gastos y margen", expanded=False):
            x, y = st.columns(2)
            with x:
                expense_pct = st.slider("Gastos como % de prima", min_value=0, max_value=50, value=25, step=1)
            with y:
                margin_pct = st.slider("Margen técnico objetivo", min_value=0, max_value=25, value=5, step=1)
            st.caption("Los resultados de cartera se calcularon con 25% de gastos y 5% de margen. Cambiar estas cargas aquí NO recalcula la simulación guardada.")
        submit_quote = st.form_submit_button("Calcular tres cotizaciones", type="primary", use_container_width=True)

    if submit_quote:
        profile = {
            "edad": int(edad), "zona": int(zona), "anios_vehiculo": int(antiguedad),
            "cobertura": int(cobertura), "historial_siniestros": int(historial),
            "tipo_uso": int(tipo_uso),
        }
        with st.spinner("Inferencia de los tres modelos congelados…"):
            try:
                quote = engine().predict(profile, expense_pct / 100, margin_pct / 100)
                st.session_state["quote_result"] = quote
                st.session_state["quote_profile"] = profile
                st.session_state["quote_charges"] = (expense_pct / 100, margin_pct / 100)
                st.session_state.pop("quote_error", None)
            except Exception as exc:
                st.session_state.pop("quote_result", None)
                st.session_state["quote_error"] = f"{type(exc).__name__}: {exc}"

    if "quote_error" in st.session_state:
        st.error("No se pudo obtener una cotización verificada. " + st.session_state["quote_error"])
        st.info("Comprueba el entorno de Python 3.11 y las versiones de requirements.txt. No se usan estimaciones sustitutas ni datos oracle como respaldo.")
    elif "quote_result" in st.session_state:
        quote = st.session_state["quote_result"]
        prev_expense, prev_margin = st.session_state["quote_charges"]
        st.markdown("#### Última cotización anual calculada")
        original_profile = st.session_state["quote_profile"]
        st.caption(
            f"Perfil cotizado: edad {original_profile['edad']} · zona {original_profile['zona']} "
            f"({zone_labels[original_profile['zona']]}) · vehículo {original_profile['anios_vehiculo']} años · "
            f"cobertura {coverage_labels[original_profile['cobertura']]} · "
            f"siniestros previos {original_profile['historial_siniestros']} · "
            f"uso {usage_labels[original_profile['tipo_uso']]}. "
            "Si cambias el formulario, vuelve a calcular para actualizar estos resultados."
        )
        cols = st.columns(3)
        for col, (_, record) in zip(cols, quote.iterrows()):
            with col:
                st.metric(NAMES[record["estrategia"]], um(record["prima_comercial"]))
                st.caption(f"Prima pura {um(record['prima_pura'])} · carga {um(record['carga_total'])}")
        chart_data = add_short_name(quote)
        chart = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=6, size=48).encode(
            x=alt.X("modelo:N", title=None, sort=list(NAMES.values())),
            y=alt.Y("prima_comercial:Q", title="Prima comercial anual (u.m.)"),
            color=alt.Color("modelo:N", scale=alt.Scale(domain=list(NAMES.values()), range=COLORS), legend=None),
            tooltip=[alt.Tooltip("modelo:N", title="Modelo"),
                     alt.Tooltip("prima_pura:Q", title="Prima pura", format=",.2f"),
                     alt.Tooltip("prima_comercial:Q", title="Prima comercial", format=",.2f")],
        ).properties(height=280)
        st.altair_chart(plot_theme(chart), use_container_width=True)
        st.dataframe(quote.set_index("estrategia").style.format({
            "prima_pura": "{:,.2f}", "prima_comercial": "{:,.2f}", "carga_total": "{:,.2f}"
        }), use_container_width=True)
        st.caption(f"Cotización independiente: gastos {pct(prev_expense, 0)} y margen objetivo {pct(prev_margin, 0)}. Unidades monetarias simuladas. No representa oferta vinculante ni prima real de mercado.")
        st.download_button("Exportar cotización CSV", data=quote.to_csv(index=False).encode("utf-8-sig"),
                           file_name="07_02_cotizacion_individual.csv", mime="text/csv")
    else:
        st.info("Completa el formulario y pulsa **Calcular tres cotizaciones** para ejecutar inferencia real con los modelos congelados.")

elif page == "Dashboard de cartera":
    st.subheader("Dashboard de cartera")
    st.markdown('<p class="section-note">Los indicadores corresponden a la misma cartera sintética y a una política de gastos del 25% más margen técnico objetivo del 5%.</p>', unsafe_allow_html=True)
    mode_label = st.radio("Método de contratación", ["Aceptaciones simuladas (U < p)", "Valor esperado (ponderación por p)"],
                          index=0, horizontal=True)
    mode = "simulated" if mode_label.startswith("Aceptaciones") else "expected"
    current = add_short_name(tables[mode]).set_index("estrategia").loc[STRATEGIES].reset_index()
    best = current.loc[current["resultado_tecnico"].idxmax()]
    avg_n = current["n_polizas"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cartera de prospectos", f"{n_prospects:,}")
    c2.metric("Estrategia de mejor resultado", NAMES[best["estrategia"]])
    c3.metric("Mejor resultado técnico", um(best["resultado_tecnico"]))
    c4.metric("Pólizas medias emitidas*", f"{avg_n:,.1f}" if mode == "expected" else f"{avg_n:,.0f}")
    st.caption("*Promedio entre alternativas contrafactuales; NO es un total agregado de pólizas distintas.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Conversión por modelo")
        chart = alt.Chart(current).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("modelo:N", title=None, sort=list(NAMES.values())),
            y=alt.Y("conversion:Q", axis=alt.Axis(format="%"), title="Conversión"),
            color=alt.Color("modelo:N", legend=None, scale=alt.Scale(domain=list(NAMES.values()), range=COLORS)),
            tooltip=["modelo:N", alt.Tooltip("conversion:Q", format=".1%")],
        ).properties(height=280)
        st.altair_chart(plot_theme(chart), use_container_width=True)
    with right:
        st.markdown("#### Resultado técnico (u.m.)")
        chart = alt.Chart(current).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("modelo:N", title=None, sort=list(NAMES.values())),
            y=alt.Y("resultado_tecnico:Q", title="Resultado técnico esperado"),
            color=alt.condition(alt.datum.resultado_tecnico >= 0, alt.value("#5CE1C4"), alt.value("#FF858F")),
            tooltip=["modelo:N", alt.Tooltip("resultado_tecnico:Q", format=",.2f")],
        ).properties(height=280)
        st.altair_chart(plot_theme(chart), use_container_width=True)

    st.markdown("#### Comparación de indicadores actuariales")
    show_cols = ["modelo", "n_polizas", "conversion", "prima_emitida", "siniestros_esperados",
                 "gastos_esperados", "resultado_tecnico", "loss_ratio", "combined_ratio", "margen_tecnico",
                 "riesgo_medio_captado", "indice_seleccion_riesgo"]
    st.dataframe(current[show_cols].style.format({
        "n_polizas": "{:,.1f}" if mode == "expected" else "{:,.0f}",
        "conversion": "{:.1%}", "prima_emitida": "{:,.0f}", "siniestros_esperados": "{:,.0f}",
        "gastos_esperados": "{:,.0f}", "resultado_tecnico": "{:,.0f}",
        "loss_ratio": "{:.1%}", "combined_ratio": "{:.1%}", "margen_tecnico": "{:.1%}",
        "riesgo_medio_captado": "{:,.1f}", "indice_seleccion_riesgo": "{:.3f}",
    }), use_container_width=True)

    with st.expander("Explorar la conversión por quintil de riesgo *oracle*", expanded=False):
        st.warning("Los quintiles se construyen con el riesgo verdadero conocido en la simulación. NO son variables del cotizador ni segmentos observables en el mercado.")
        segments = add_short_name(tables["segments"])
        chart = alt.Chart(segments).mark_bar().encode(
            x=alt.X("segmento:N", sort=["Q1 · bajo", "Q2", "Q3", "Q4", "Q5 · alto"], title="Quintil de riesgo simulado"),
            y=alt.Y("conversion:Q", axis=alt.Axis(format="%"), title="Conversión simulada"),
            xOffset=alt.XOffset("modelo:N", sort=list(NAMES.values())),
            color=alt.Color("modelo:N", scale=alt.Scale(domain=list(NAMES.values()), range=COLORS)),
            tooltip=["segmento:N", "modelo:N", alt.Tooltip("conversion:Q", format=".1%")],
        ).properties(height=330)
        st.altair_chart(plot_theme(chart), use_container_width=True)
    st.divider()
    st.markdown("### Análisis por segmentos observables")
    st.caption(
        "Corrida base congelada 07.01 · 1,500 prospectos · L = 1.50, k = 5. "
        "Los indicadores comerciales de esta sección son SIEMPRE valores esperados "
        "(ponderados por probabilidad de contratación), independientemente del modo "
        "seleccionado arriba. No responden a los controles de Sensibilidad de demanda."
    )

    metric_options = {
        "Sesgo técnico relativo (todos los prospectos)": (
            "sesgo_relativo_todos", "Sesgo relativo frente al oracle", ".1%",
            "El sesgo es (suma de prima pura estimada − suma oracle) / suma oracle "
            "en TODOS los prospectos. Negativo significa subestimación; no implica "
            "automáticamente resultado técnico negativo.", 0.0,
        ),
        "MAE de prima pura (todos los prospectos)": (
            "mae_oracle_todos", "MAE de prima pura (u.m. por prospecto)", ",.0f",
            "Error absoluto medio de prima pura contra oracle, calculado sobre TODOS "
            "los prospectos del segmento, no solo los que aceptarían.", None,
        ),
        "Conversión comercial esperada": (
            "conversion_esperada", "Contratación esperada / prospectos", ".1%",
            "Conversión = suma de probabilidades de contratación / n prospectos. "
            "También depende del precio y de la demanda hipotética: menor conversión "
            "NO prueba sobreestimación del riesgo.", None,
        ),
        "Prima comercial media ofertada": (
            "precio_medio_ofertado", "Precio medio ofertado (u.m. por prospecto)", ",.0f",
            "Promedio simple del precio comercial ofrecido a TODOS los prospectos, "
            "con gastos 25% y margen objetivo 5%; no es la prima media de los contratantes.", None,
        ),
        "Resultado técnico esperado": (
            "resultado_tecnico_esperado", "Resultado técnico esperado total (u.m.)", ",.0f",
            "Suma de prima emitida esperada − gastos esperados − siniestros esperados "
            "en cada segmento. Es un TOTAL, influido por el tamaño y la selección "
            "de la cartera; no corresponde a un resultado contable observado.", 0.0,
        ),
        "Siniestralidad esperada (loss ratio)": (
            "loss_ratio_esperado", "Loss ratio esperado", ".1%",
            "Siniestros esperados / prima emitida esperada de los contratantes "
            "ponderados. Excluye gastos; debe leerse junto con el combined ratio.", 1.0,
        ),
        "Combined ratio esperado": (
            "combined_ratio_esperado", "Combined ratio esperado", ".1%",
            "(Siniestros esperados + gastos esperados) / prima emitida esperada. "
            "Más de 100% implica pérdida técnica esperada, bajo estos supuestos.", 1.0,
        ),
    }
    dim_col, measure_col = st.columns([1, 2])
    with dim_col:
        dimension = st.selectbox(
            "Variable observable", options=list(SEGMENT_ORDER), key="observable_dimension",
        )
    with measure_col:
        metric_label = st.selectbox(
            "Indicador a comparar", options=list(metric_options), key="observable_metric",
        )
    field, axis_title, value_format, explanation, reference = metric_options[metric_label]
    segment_view = add_short_name(
        tables["observable_segments"].loc[
            tables["observable_segments"]["dimension"] == dimension
        ]
    )
    st.caption(
        f"{dimension}: {len(SEGMENT_ORDER[dimension])} grupos predefinidos, "
        f"{n_prospects:,} prospectos por modelo (mismos perfiles en tres estrategias "
        "contrafactuales). Las dimensiones no se suman entre sí."
    )

    bars = alt.Chart(segment_view).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("segmento:N", title=dimension, sort=SEGMENT_ORDER[dimension]),
        xOffset=alt.XOffset("modelo:N", sort=list(NAMES.values())),
        y=alt.Y(f"{field}:Q", title=axis_title,
                axis=alt.Axis(format=value_format)),
        color=alt.Color(
            "modelo:N", title="Modelo",
            scale=alt.Scale(domain=list(NAMES.values()), range=COLORS),
        ),
        tooltip=[
            alt.Tooltip("segmento:N", title="Segmento"),
            alt.Tooltip("modelo:N", title="Modelo"),
            alt.Tooltip("n_prospectos:Q", title="Prospectos", format=",.0f"),
            alt.Tooltip(f"{field}:Q", title=axis_title, format=value_format),
            alt.Tooltip("conversion_esperada:Q", title="Conversión esperada", format=".1%"),
            alt.Tooltip("combined_ratio_esperado:Q", title="Combined ratio", format=".1%"),
        ],
    )
    if reference is not None:
        guideline = alt.Chart(pd.DataFrame({"ref": [reference]})).mark_rule(
            color="#D0DCE9", strokeDash=[5, 5], strokeWidth=1.2,
        ).encode(y="ref:Q")
        segment_chart = alt.layer(bars, guideline)
    else:
        segment_chart = bars
    st.altair_chart(plot_theme(segment_chart.properties(height=340)), use_container_width=True)
    st.caption(explanation + (" Línea discontinua = referencia (0 o 100%)."
                              if reference is not None else ""))

    # Tabla corta: tamaño y una lectura simultánea de riesgo y suficiencia.
    visible_columns = list(dict.fromkeys([
        "segmento", "modelo", "n_prospectos", field,
        "conversion_esperada", "precio_medio_ofertado",
        "resultado_tecnico_esperado", "combined_ratio_esperado",
    ]))
    column_labels = {
        "segmento": "Segmento", "modelo": "Modelo", "n_prospectos": "Prospectos",
        "sesgo_relativo_todos": "Sesgo vs oracle",
        "mae_oracle_todos": "MAE (u.m.)",
        "conversion_esperada": "Conversión esperada",
        "precio_medio_ofertado": "Precio medio (u.m.)",
        "resultado_tecnico_esperado": "Resultado técnico (u.m.)",
        "loss_ratio_esperado": "Siniestralidad esperada",
        "combined_ratio_esperado": "Combined ratio",
    }
    table_formats = {
        "Prospectos": "{:,.0f}", "Sesgo vs oracle": "{:.1%}",
        "MAE (u.m.)": "{:,.0f}", "Conversión esperada": "{:.1%}",
        "Precio medio (u.m.)": "{:,.0f}", "Resultado técnico (u.m.)": "{:,.0f}",
        "Siniestralidad esperada": "{:.1%}", "Combined ratio": "{:.1%}",
    }
    display_segments = segment_view[visible_columns].rename(columns=column_labels)
    st.dataframe(
        display_segments.style.format({k: v for k, v in table_formats.items()
                                       if k in display_segments.columns}),
        use_container_width=True, hide_index=True,
    )
    with st.expander("Ver métricas adicionales y alcance metodológico", expanded=False):
        st.markdown(
            "**Técnica vs. comercial:** los sesgos y el MAE comparan prima pura "
            "estimada con oracle en todos los prospectos; los precios son ofertas "
            "comerciales y la rentabilidad se pondera por las probabilidades de compra. "
            "La prima media de compradores esperados difiere de la oferta media."
        )
        st.markdown(
            "**Interpretación:** dimensiones marginales y superpuestas; resultados "
            "descriptivos, no causales ni ajustados. La demanda simulada utiliza oracle "
            "para generar WTP, pero el oracle nunca ingresa al cotizador. Los grupos "
            "pequeños deben interpretarse con cautela."
        )
        extra_cols = [
            "segmento", "modelo", "n_prospectos", "prima_pura_oracle_media",
            "prima_pura_estimada_media", "sesgo_relativo_todos", "mae_oracle_todos",
            "sesgo_relativo_aceptados_esperados", "conversion_simulada",
            "conversion_esperada", "precio_medio_ofertado",
            "precio_medio_aceptados_esperados", "contrataciones_esperadas",
            "resultado_tecnico_esperado", "loss_ratio_esperado", "combined_ratio_esperado",
        ]
        st.dataframe(segment_view[extra_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Exportar métricas de la segmentación seleccionada",
        data=segment_view.drop(columns=["modelo"]).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"07_02_segmentacion_{list(SEGMENT_ORDER).index(dimension) + 1}.csv",
        mime="text/csv", key="segment_download",
    )
    st.warning(
        "Este análisis corresponde exclusivamente al escenario base congelado. "
        "La elección de modo del dashboard y los controles L/k no modifican "
        "los CSV de segmentación. La menor conversión no equivale automáticamente "
        "a sobreestimación del riesgo."
    )

    st.download_button("Descargar tabla de KPI", data=tables[mode].to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"07_02_kpi_{mode}.csv", mime="text/csv")

elif page == "Sensibilidad de demanda":
    st.subheader("Sensibilidad de demanda")
    st.markdown('<p class="section-note">Se comparan nueve escenarios hipotéticos de disposición a pagar, sin volver a ejecutar los modelos. Todos emplean la misma cartera y sus precios originales.</p>', unsafe_allow_html=True)
    sens = add_short_name(tables["sensitivity"])
    l_levels = sorted(sens["L"].unique().tolist())
    k_levels = sorted(sens["k"].unique().tolist())
    l_col, k_col = st.columns(2)
    with l_col:
        L = st.select_slider("Factor hipotético de disposición a pagar (L)", options=l_levels, value=1.5,
                             format_func=lambda x: f"{x:.2f}")
    with k_col:
        k = st.select_slider("Sensibilidad a precio (k)", options=k_levels, value=5.0,
                             format_func=lambda x: f"{x:g}")
    chosen = sens.loc[np.isclose(sens["L"], L) & np.isclose(sens["k"], k)].sort_values("resultado_tecnico", ascending=False)
    winner = chosen.iloc[0]
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Mayor resultado técnico", NAMES[winner["estrategia"]])
    cc2.metric("Resultado esperado del líder", um(winner["resultado_tecnico"]))
    cc3.metric("Conversión del líder", pct(winner["conversion"]))
    st.markdown("#### Comparación en el escenario elegido")
    st.dataframe(chosen[["modelo", "n_polizas", "conversion", "prima_emitida", "resultado_tecnico",
                         "combined_ratio", "margen_tecnico"]].style.format({
        "n_polizas": "{:,.1f}", "conversion": "{:.1%}", "prima_emitida": "{:,.0f}",
        "resultado_tecnico": "{:,.0f}", "combined_ratio": "{:.1%}", "margen_tecnico": "{:.1%}",
    }), use_container_width=True)

    st.markdown("#### Resultado técnico esperado en los nueve escenarios")

        # Escala global compartida para los tres modelos
    global_min = float(sens["resultado_tecnico"].min())
    global_max = float(sens["resultado_tecnico"].max())

    shared_color = alt.Scale(
        domain=[global_min, 0, global_max],
        scheme="redyellowgreen",
        clamp=True,
    )

    for model_name in list(NAMES.values()):
        current = sens[sens["modelo"] == model_name].copy()

        base = alt.Chart(current).encode(
            x=alt.X(
                "k:O",
                title="Sensibilidad de demanda k",
                sort=k_levels
            ),
            y=alt.Y(
                "L:O",
                title="Factor WTP L",
                sort=l_levels
            )
        )

        heat = base.mark_rect(cornerRadius=3).encode(
            color=alt.Color(
                "resultado_tecnico:Q",
                title="Resultado técnico (u.m.)",
                scale=shared_color,
                legend=alt.Legend(format=",.0f")
            ),
            tooltip=[
                "modelo:N",
                alt.Tooltip("L:Q", format=".2f"),
                alt.Tooltip("k:Q", format=".1f"),
                alt.Tooltip("resultado_tecnico:Q", format=",.0f"),
                alt.Tooltip("conversion:Q", format=".1%"),
            ],
        )

        text = base.mark_text(
            fontSize=15,
            fontWeight="bold",
            color="white",
            stroke="black",
            strokeWidth=0.6
        ).encode(
            text=alt.Text("resultado_tecnico:Q", format=",.0f")
        )

        chart = (heat + text).properties(
            width=520,
            height=300,
            title=model_name,
        )

        st.altair_chart(chart, use_container_width=True)
    st.markdown("#### Estabilidad del resultado por modelo")
    stats = sens.groupby(["estrategia", "modelo"], as_index=False).agg(
        escenarios_positivos=("resultado_tecnico", lambda series: int((series > 0).sum())),
        resultado_min=("resultado_tecnico", "min"), resultado_max=("resultado_tecnico", "max"),
    )
    counts = tables["winners"]["estrategia"].value_counts()
    stats["escenarios_ganados"] = stats["estrategia"].map(counts).fillna(0).astype(int)
    st.dataframe(stats[["modelo", "escenarios_ganados", "escenarios_positivos", "resultado_min", "resultado_max"]]
                 .style.format({"resultado_min": "{:,.0f}", "resultado_max": "{:,.0f}"}), use_container_width=True)
    st.caption("WTP = prima pura oracle × L × exp(0.32 × Z). Contratación esperada = función logística de k × ln(WTP/precio). Z y los pronósticos están congelados. Estos parámetros no fueron calibrados con ventas reales.")
    st.download_button("Exportar sensibilidad completa", data=tables["sensitivity"].to_csv(index=False).encode("utf-8-sig"),
                       file_name="07_02_sensibilidad.csv", mime="text/csv")

else:
    st.subheader("Metodología y límites")
    st.markdown("""
    **Pregunta experimental:** ¿el modelo con mejor predicción de riesgo produce también una cartera
    más rentable bajo una política comercial común?

    **Estimadores congelados:** GLM Tweedie directo (estadístico), XGBoost directo (ML) y
    MLP Tweedie directo (DL). Los representantes se eligieron en la fase 06 bajo métricas
    distintas, antes de evaluar sus resultados comerciales.

    **Entradas de inferencia:** edad, zona, antigüedad del vehículo, cobertura,
    historial de siniestros y tipo de uso. Los campos del *oracle* no ingresan al cotizador.
    """)
    st.latex(r"\widehat{P}_i = \frac{\widehat{PP}_i}{1-g-m} \quad (g=0.25,\;m=0.05)")
    st.markdown("""
    **Precio:** la misma fórmula para los tres estimadores, con gastos variables como
    proporción de la prima comercial y margen técnico objetivo también expresado sobre ella.
    Cambiar los controles de cargas solo modifica la cotización individual, no los resultados exportados.
    """)
    st.latex(r"W_i(L)=PP_i^*\,L\,e^{0.32Z_i},\qquad q_i=\frac{1}{1+\exp\!\left[k\,\log(P_i/W_i)\right]}")
    st.markdown("""
    **Demanda:** la disposición a pagar es hipotética y depende del *oracle* únicamente dentro
    del experimento. Las compras simuladas se obtienen al comparar un uniforme común con
    la probabilidad de aceptación de cada precio. El modo ponderado usa directamente esas probabilidades.
    """)
    st.latex(r"\Pi=\sum_i w_i\,P_i-0.25\sum_i w_i\,P_i-\sum_i w_i\,PP_i^*")
    st.markdown("""
    Donde `w_i` representa la aceptación binaria simulada o la probabilidad de compra.
    El resultado técnico es **esperado respecto a los siniestros**, no utilidad contable
    observada. El *combined ratio* se calcula como `(siniestros esperados + gastos) / prima emitida`.

    **Limitaciones:** cartera sintética, exposición anual unitaria, sin impuestos ni comisiones
    adicionales, sin calibración de demanda real y sin análisis de restricciones regulatorias.
    Ninguna cotización equivale a una oferta real de seguro.

    **Reproducibilidad:** las seis tablas guardadas de 07.01 se validan al abrir la app;
    el motor no reentrena, ni actualiza, ni sustituye los modelos persistidos.
    """)
    st.info("En 07.01 el GLM lideró los nueve escenarios de sensibilidad de demanda; esa conclusión aplica solamente a los supuestos y prospectos de este ejercicio.")

st.markdown("<div class='footnote'>PurePremium Lab · Demostración académica con datos sintéticos · Cifras en unidades monetarias simuladas (u.m.)</div>",
            unsafe_allow_html=True)
