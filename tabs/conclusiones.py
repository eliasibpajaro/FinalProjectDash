from dash import html
import dash_bootstrap_components as dbc


def layout():
    return dbc.Container(
        [
            html.Div("Cierre técnico", className="section-kicker"),
            html.H1("Conclusiones", className="page-title"),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(
                            "El dashboard consolida un flujo reproducible para analizar y predecir aceleraciones espectrales "
                            "a partir de modelos residuales. La arquitectura modular permite mantener cada pestaña en un archivo "
                            "independiente, facilitando mantenimiento, expansión y despliegue.",
                        ),
                        html.P(
                            "El uso de GRU es coherente con la naturaleza secuencial del espectro: cada registro se representa "
                            "como una secuencia ordenada de períodos, y la red aprende una corrección residual dependiente del período.",
                        ),
                        html.P(
                            "La validación LOGO por evento permite evaluar mejor la capacidad de generalización ante sismos no vistos. "
                            "Las métricas por período ayudan a identificar en qué rangos espectrales la corrección residual aporta mayor reducción de error.",
                            className="mb-0",
                        ),
                    ]
                ),
                className="highlight-card",
            ),
        ],
        fluid=True,
    )