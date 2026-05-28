from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from model.train_model import (
    PERIOD_COL,
    SA_COL,
    TMAX_COL,
    NOSAM_RESIDUAL_COL,
    standardize_columns,
    read_table,
)


def load_problem_data():
    try:
        df = read_table("Resids_for_Eliasib.xlsx")
        df = standardize_columns(df)

        df[PERIOD_COL] = pd.to_numeric(df[PERIOD_COL], errors="coerce")
        df[TMAX_COL] = pd.to_numeric(df[TMAX_COL], errors="coerce")

        if SA_COL in df.columns:
            df[SA_COL] = pd.to_numeric(df[SA_COL], errors="coerce")

        if NOSAM_RESIDUAL_COL in df.columns:
            df[NOSAM_RESIDUAL_COL] = pd.to_numeric(df[NOSAM_RESIDUAL_COL], errors="coerce")

        return df

    except Exception as exc:
        return exc


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15),
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_visible=False,
        yaxis_visible=False,
    )
    return fig


def period_options(df):
    periods = sorted(pd.to_numeric(df[PERIOD_COL], errors="coerce").dropna().unique())
    return [{"label": f"T = {p:g} s", "value": float(p)} for p in periods]


def residual_histogram(df, selected_period, only_tmax=True):
    if NOSAM_RESIDUAL_COL not in df.columns:
        return empty_figure("No existe la columna Total con residuales en la base.")

    dff = df[
        np.isclose(
            df[PERIOD_COL].astype(float),
            float(selected_period),
            rtol=0,
            atol=1e-10,
        )
    ].copy()

    if only_tmax:
        dff = dff[dff[PERIOD_COL] <= dff[TMAX_COL]]

    dff = dff[np.isfinite(dff[NOSAM_RESIDUAL_COL])]

    if dff.empty:
        return empty_figure(f"No hay residuales válidos para T={selected_period:g} s.")

    fig = px.histogram(
        dff,
        x=NOSAM_RESIDUAL_COL,
        nbins=45,
        marginal="box",
        title=f"Distribución de residuales base NoSAm — T={selected_period:g} s",
        labels={
            NOSAM_RESIDUAL_COL: "Residual ln(Sa_obs) − ln(Sa_GMPE)",
            "count": "Frecuencia",
        },
    )

    fig.add_vline(
        x=float(dff[NOSAM_RESIDUAL_COL].mean()),
        line_dash="dash",
        annotation_text="media",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        bargap=0.03,
    )

    return fig


def lnsa_histogram(df, selected_period, only_tmax=True):
    if SA_COL not in df.columns:
        return empty_figure("No existe la columna Sa en la base.")

    dff = df[
        np.isclose(
            df[PERIOD_COL].astype(float),
            float(selected_period),
            rtol=0,
            atol=1e-10,
        )
    ].copy()

    if only_tmax:
        dff = dff[dff[PERIOD_COL] <= dff[TMAX_COL]]

    dff["ln_Sa"] = np.log(pd.to_numeric(dff[SA_COL], errors="coerce").clip(lower=1e-12))
    dff = dff[np.isfinite(dff["ln_Sa"])]

    if dff.empty:
        return empty_figure(f"No hay Sa válido para T={selected_period:g} s.")

    fig = px.histogram(
        dff,
        x="ln_Sa",
        nbins=45,
        marginal="box",
        title=f"Distribución de ln(Sa) observado — T={selected_period:g} s",
        labels={"ln_Sa": "ln(Sa)", "count": "Frecuencia"},
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        bargap=0.03,
    )

    return fig


def layout():
    data = load_problem_data()

    if isinstance(data, Exception):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5("No se pudo cargar la base principal.", className="alert-heading"),
                    html.P(str(data)),
                    html.P("Coloca Resids_for_Eliasib.xlsx en la raíz del proyecto."),
                ],
                color="warning",
            ),
            fluid=True,
        )

    df = data.copy()
    opts = period_options(df)
    default_period = opts[0]["value"] if opts else None

    valid_by_period = (
        df.assign(is_valid=df[PERIOD_COL] <= df[TMAX_COL])
        .groupby(PERIOD_COL, as_index=False)["is_valid"]
        .mean()
        .rename(columns={"is_valid": "fraction_valid"})
    )

    fig_mask = px.line(
        valid_by_period,
        x=PERIOD_COL,
        y="fraction_valid",
        markers=True,
        title="Fracción de valores válidos por período usando Tmax",
        labels={PERIOD_COL: "Período T (s)", "fraction_valid": "Fracción válida"},
    )
    fig_mask.update_xaxes(type="log")
    fig_mask.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))

    summary_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3(f"{df['Record Sequence Number'].nunique():,}"),
                            html.P("registros"),
                        ]
                    ),
                    className="metric-card",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3(f"{df['EQID_Code'].nunique():,}"),
                            html.P("eventos"),
                        ]
                    ),
                    className="metric-card",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3(f"{df[PERIOD_COL].nunique():,}"),
                            html.P("períodos"),
                        ]
                    ),
                    className="metric-card",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3(f"{df['Station Code'].nunique():,}"),
                            html.P("estaciones"),
                        ]
                    ),
                    className="metric-card",
                ),
                md=3,
            ),
        ],
        className="g-4 mb-4",
    )

    return dbc.Container(
        [
            html.Div("Diagnóstico de la variable objetivo", className="section-kicker"),
            html.H1("Comportamiento por período de Sa, residuales y máscara Tmax", className="page-title"),
            html.P(
                "Como la base está en formato largo, las distribuciones deben revisarse período por período. "
                "El selector evita mezclar residuales de diferentes ordenadas espectrales.",
                className="lead",
            ),
            summary_cards,
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Período espectral"),
                                    dcc.Dropdown(
                                        id="problem-period",
                                        options=opts,
                                        value=default_period,
                                        clearable=False,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Checklist(
                                    id="problem-use-tmax",
                                    options=[{"label": "Aplicar máscara Tmax", "value": "tmax"}],
                                    value=["tmax"],
                                    switch=True,
                                    className="mt-4",
                                ),
                                md=4,
                            ),
                        ],
                        className="g-3",
                    )
                ),
                className="soft-card mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="problem-residual-hist")),
                            className="soft-card",
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="problem-lnsa-hist")),
                            className="soft-card",
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(figure=fig_mask)),
                            className="soft-card",
                        ),
                        lg=12,
                    ),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )


def register_callbacks(app):
    @app.callback(
        Output("problem-residual-hist", "figure"),
        Output("problem-lnsa-hist", "figure"),
        Input("problem-period", "value"),
        Input("problem-use-tmax", "value"),
    )
    def update_problem_histograms(period, use_tmax):
        df = load_problem_data()

        if isinstance(df, Exception) or period is None:
            fig = empty_figure("No se pudo cargar la base.")
            return fig, fig

        only_tmax = "tmax" in (use_tmax or [])

        return (
            residual_histogram(df, float(period), only_tmax),
            lnsa_histogram(df, float(period), only_tmax),
        )