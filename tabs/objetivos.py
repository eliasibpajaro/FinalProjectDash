from dash import html
import dash_bootstrap_components as dbc


def layout():
    objectives = [
        ("Objetivo general", "Desarrollar una aplicación analítica que integre EDA, validación y predicción de aceleraciones espectrales mediante modelos residuales basados en GRU."),
        ("Objetivo específico 1", "Preparar el dataset largo, transformar variables predictoras y construir matrices espectrales respetando la máscara Tmax."),
        ("Objetivo específico 2", "Entrenar modelos residuales comparables bajo validación LOGO por evento para evaluar generalización sísmica."),
        ("Objetivo específico 3", "Visualizar métricas por período: SD, RMSE, MAE, MSE, R², reales vs predichos y distribución de residuos."),
        ("Objetivo específico 4", "Permitir predicción interactiva para escenarios definidos por magnitud, distancia, profundidad, clase de suelo y variables regionales."),
    ]

    return dbc.Container(
        [
            html.Div("Propósito metodológico", className="section-kicker"),
            html.H1("Objetivos y justificación", className="page-title"),
            html.P(
                "La aplicación organiza el flujo de investigación en una herramienta reproducible: carga de datos, "
                "entrenamiento, validación cruzada por evento y predicción interactiva.",
                className="lead",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(title, className="mb-3"),
                                    html.P(text, className="mb-0"),
                                ]
                            ),
                            className="soft-card h-100",
                        ),
                        md=6,
                    )
                    for title, text in objectives
                ],
                className="g-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Justificación", className="mb-3"),
                        html.P(
                            "El aprendizaje residual es útil cuando se desea preservar la estructura de una GMPE y, al mismo tiempo, "
                            "capturar patrones regionales o sistemáticos que no quedan plenamente representados por el modelo base. "
                            "La validación LOGO por evento evita que registros del mismo sismo aparezcan simultáneamente en entrenamiento "
                            "y validación, por lo que es más exigente que una partición aleatoria por registro.",
                            className="mb-0",
                        ),
                    ]
                ),
                className="highlight-card mt-4",
            ),
        ],
        fluid=True,
    )