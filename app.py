import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent

for path in [PROJECT_DIR, PARENT_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

from tabs import (
    introduccion,
    problema,
    objetivos,
    resultados,
    mapa_interactivo,
    prediccion,
    comparacion_ask14_nosam,
    limitaciones,
    conclusiones,
)


external_stylesheets = [
    dbc.themes.FLATLY,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
]


app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="Predicción de Espectros de Respuesta",
)

server = app.server


# =========================================================
# MÓDULOS DE PESTAÑAS
# =========================================================

TAB_MODULES = {
    "introduccion": introduccion,
    "problema": problema,
    "objetivos": objetivos,
    "resultados": resultados,
    "mapa_interactivo": mapa_interactivo,
    "prediccion": prediccion,
    "comparacion_ask14_nosam": comparacion_ask14_nosam,
    "limitaciones": limitaciones,
    "conclusiones": conclusiones,
}

# =========================================================
# NAVBAR
# =========================================================

def make_navbar():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div("Sa", className="brand-badge"),
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    "Dashboard de Aceleraciones Espectrales",
                                    className="brand-title",
                                ),
                                html.Div(
                                    "Aprendizaje residual + GRU + validación LOGO",
                                    className="brand-subtitle",
                                ),
                            ],
                            width=True,
                        ),
                    ],
                    align="center",
                    className="g-2",
                )
            ],
            fluid=True,
        ),
        className="top-navbar",
        dark=False,
    )


# =========================================================
# TABS
# =========================================================

def make_tabs():
    return dbc.Card(
        dbc.CardBody(
            dcc.Tabs(
                id="main-tabs",
                value="introduccion",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        label="Introducción",
                        value="introduccion",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="Problema",
                        value="problema",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="Objetivos",
                        value="objetivos",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="Resultados",
                        value="resultados",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="Mapa interactivo",
                        value="mapa_interactivo",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="Predicción",
                        value="prediccion",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="ASK14 residual vs NoSAm",
                        value="comparacion_ask14_nosam",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="Limitaciones",
                        value="limitaciones",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                    dcc.Tab(
                        label="Conclusiones",
                        value="conclusiones",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                    ),
                ],
            )
        ),
        className="tabs-card",
    )


# =========================================================
# LAYOUT PRINCIPAL
# =========================================================

app.layout = html.Div(
    [
        make_navbar(),
        dbc.Container(
            [
                make_tabs(),
                html.Div(
                    id="tab-content",
                    className="tab-content-wrapper",
                ),
            ],
            fluid=True,
            className="main-container",
        ),
    ]
)


# =========================================================
# ROUTER DE PESTAÑAS
# =========================================================

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab_name):
    module = TAB_MODULES.get(tab_name, introduccion)
    return module.layout()


# =========================================================
# REGISTRO DESACOPLADO DE CALLBACKS
# =========================================================

for module in TAB_MODULES.values():
    if hasattr(module, "register_callbacks"):
        module.register_callbacks(app)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)