from dash import html
import dash_bootstrap_components as dbc


def metric_card(title, text, icon):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(className=f"fa-solid {icon} card-icon"),
                html.H4(title, className="card-title"),
                html.P(text, className="card-text"),
            ]
        ),
        className="soft-card h-100",
    )


def layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Ingeniería sísmica + ciencia de datos", className="section-kicker"),
                            html.H1("Predicción de aceleraciones espectrales mediante aprendizaje residual", className="page-title"),
                            html.P(
                                "Este dashboard resume un flujo de modelación para espectros de respuesta RotD50. "
                                "La idea central es partir de un modelo físico o empírico base, como NoSAm o ASK14, "
                                "y entrenar una red GRU para aprender el residuo logarítmico entre el espectro observado "
                                "y el espectro estimado por la GMPE.",
                                className="lead",
                            ),
                        ],
                        lg=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Arquitectura del enfoque", className="mb-3"),
                                    html.Ul(
                                        [
                                            html.Li("Entrada: magnitud, distancia Rrup, profundidad hipocentral, clase de suelo y variables regionales."),
                                            html.Li("Salida: secuencia de residuos o espectro completo en 22 períodos."),
                                            html.Li("Máscara: Tmax limita los períodos confiables por registro."),
                                            html.Li("Validación: Leave-One-Group-Out por evento sísmico."),
                                        ],
                                        className="mb-0",
                                    ),
                                ]
                            ),
                            className="highlight-card",
                        ),
                        lg=4,
                    ),
                ],
                className="g-4 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(metric_card("Espectro de respuesta", "Representa la demanda máxima esperada de un oscilador SDOF para diferentes períodos.", "fa-wave-square"), md=4),
                    dbc.Col(metric_card("Residual learning", "La GRU no reemplaza la GMPE: aprende la corrección residual en escala logarítmica.", "fa-brain"), md=4),
                    dbc.Col(metric_card("Máscara Tmax", "Ignora períodos no válidos o poco confiables durante entrenamiento y evaluación.", "fa-filter"), md=4),
                ],
                className="g-4",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Interpretación del target", className="mb-3"),
                                html.P(
                                    "Para los modelos residuales, el objetivo principal es el residuo logarítmico: "
                                    "residuo = ln(Sa observado) - ln(Sa GMPE). "
                                    "Luego, para predecir el espectro corregido, se combina la GMPE con la predicción residual: "
                                    "ln(Sa corregido) = ln(Sa GMPE) + residuo_predicho.",
                                    className="mb-0",
                                ),
                            ]
                        ),
                        className="soft-card mt-4",
                    ),
                    width=12,
                )
            ),
        ],
        fluid=True,
    )