from dash import html
import dash_bootstrap_components as dbc


def layout():
    limitations = [
        ("Tmax y disponibilidad espectral", "Los períodos mayores que Tmax se excluyen del entrenamiento y de la evaluación. Esto reduce información efectiva en períodos largos."),
        ("Generalización por evento", "LOGO por evento es exigente: algunos eventos tienen pocos registros, lo que puede aumentar la variabilidad de las métricas fold a fold."),
        ("Dependencia de la GMPE base", "El modelo residual corrige una GMPE; por tanto, la calidad de la predicción final también depende de la consistencia del modelo base."),
        ("ASK14 requiere OpenQuake", "El entrenamiento o predicción de ASK14 necesita instalar OpenQuake. Si no está disponible, el dashboard puede operar con los modelos NoSAm."),
        ("Extrapolación", "Predicciones fuera del dominio de magnitud, distancia, profundidad o clase de suelo observados deben interpretarse con cautela."),
    ]

    return dbc.Container(
        [
            html.Div("Alcance del sistema", className="section-kicker"),
            html.H1("Limitaciones", className="page-title"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(title),
                                    html.P(text, className="mb-0"),
                                ]
                            ),
                            className="soft-card h-100",
                        ),
                        md=6,
                    )
                    for title, text in limitations
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )