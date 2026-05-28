from functools import lru_cache
from pathlib import Path
import re

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

try:
    import dash_leaflet as dl
except Exception as exc:
    dl = None
    DL_IMPORT_ERROR = exc


# =========================================================
# CONFIGURACIÓN
# =========================================================

DATA_FILE = "Resids_for_Eliasib.xlsx"


MAP_TILES = {
    "openstreet": {
        "label": "OpenStreetMap",
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "&copy; OpenStreetMap contributors",
    },
    "positron": {
        "label": "Carto Positron",
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "carto": {
        "label": "Carto Voyager",
        "url": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "topografico": {
        "label": "Topográfico / relieve",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles &copy; Esri",
    },
}


CANONICAL_COLUMNS = {
    "event_id": [
        "EQID_Code",
        "EQID",
        "Event ID",
        "Event_ID",
        "event_id",
    ],
    "station_code": [
        "Station Code",
        "Station_Code",
        "Station",
        "station_code",
    ],
    "event_lat": [
        "Epicenter Latitude",
        "Epicenter Latitude (deg positive N)",
        "Epicenter_Latitude",
        "Event Latitude",
        "lat_event",
    ],
    "event_lon": [
        "Epicenter Longitude",
        "Epicenter Longitude (deg positive E)",
        "Epicenter_Longitude",
        "Event Longitude",
        "lon_event",
    ],
    "station_lat": [
        "Station Latitude",
        "Station Latitude (deg positive N)",
        "Station_Latitude",
        "lat_station",
    ],
    "station_lon": [
        "Station Longitude",
        "Station Longitude (deg positive E)",
        "Station_Longitude",
        "lon_station",
    ],
    "magnitude": [
        "Magnitude",
        "Mw",
        "mag",
        "magnitude",
    ],
    "zhypo_km": [
        "Hypocenter Depth (km)",
        "Hypocentral Depth (km)",
        "Depth",
        "Depth_km",
        "zhypo",
        "Zhypo",
    ],
    "rrup_km": [
        "Rrup_OpenQuake",
        "Rrup",
        "Rrup (km)",
        "rrup",
        "rrup_openquake",
    ],
    "tmax": [
        "Tmax",
        "T_max",
        "TMAX",
        "T max",
        "T corner",
        "Tcorner",
    ],
    "soil_class": [
        "Soil_Class",
        "Soil Class",
        "cat",
        "Cat",
        "soil_class",
        "Clase Suelo",
        "Clase de suelo",
    ],
}


# =========================================================
# UTILIDADES DE DATOS
# =========================================================

def normalize_colname(name):
    text = str(name).strip().lower()
    text = text.replace("á", "a").replace("é", "e").replace("í", "i")
    text = text.replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_column(df, candidates, required=True):
    normalized = {normalize_colname(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in normalized:
            return normalized[key]

    for candidate in candidates:
        key = normalize_colname(candidate)
        for norm_col, real_col in normalized.items():
            if key and key in norm_col:
                return real_col

    if required:
        raise ValueError(
            "No se encontró una columna requerida. "
            f"Candidatas esperadas: {candidates}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    return None


def resolve_data_path():
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent

    candidates = [
        project_dir / DATA_FILE,
        Path.cwd() / DATA_FILE,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No se encontró {DATA_FILE}. Debe estar en la raíz del proyecto, al mismo nivel de app.py."
    )


@lru_cache(maxsize=1)
def load_map_data():
    path = resolve_data_path()
    raw = pd.read_excel(path)

    colmap = {}

    for canonical, candidates in CANONICAL_COLUMNS.items():
        required = canonical not in ["event_id", "station_code", "soil_class", "tmax"]
        colmap[canonical] = find_column(raw, candidates, required=required)

    df = pd.DataFrame()

    for canonical, source_col in colmap.items():
        if source_col is not None:
            df[canonical] = raw[source_col]
        else:
            df[canonical] = None

    if df["event_id"].isna().all():
        df["event_id"] = raw.index.astype(str)

    if df["station_code"].isna().all():
        df["station_code"] = raw.index.astype(str)

    df["event_id"] = df["event_id"].astype(str)
    df["station_code"] = df["station_code"].astype(str)

    numeric_cols = [
        "event_lat",
        "event_lon",
        "station_lat",
        "station_lon",
        "magnitude",
        "zhypo_km",
        "rrup_km",
        "tmax",
        "soil_class",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["tmax"].isna().all():
        df["tmax"] = 0.0

    df = df.dropna(
        subset=[
            "event_lat",
            "event_lon",
            "station_lat",
            "station_lon",
            "magnitude",
            "zhypo_km",
            "rrup_km",
        ]
    ).copy()

    df["soil_class"] = df["soil_class"].fillna(-1).astype(int)

    # La base está en formato largo. Para mapa se trabaja por par evento-estación.
    # Si una estación-evento aparece en varios períodos, queda una fila espacial.
    df = df.drop_duplicates(
        subset=[
            "event_id",
            "station_code",
            "event_lat",
            "event_lon",
            "station_lat",
            "station_lon",
        ]
    ).reset_index(drop=True)

    return df


def get_bounds(series, pad=0.0):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return 0.0, 1.0

    vmin = float(s.min())
    vmax = float(s.max())

    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    if pad > 0:
        delta = vmax - vmin
        vmin -= delta * pad
        vmax += delta * pad

    return vmin, vmax


def make_marks(vmin, vmax, n=5):
    values = np.linspace(vmin, vmax, n)
    marks = {}

    for v in values:
        key = float(round(v, 3))
        marks[key] = f"{v:.2g}"

    return marks


def make_range_slider(slider_id, label, vmin, vmax, step=None):
    if step is None:
        step = max((vmax - vmin) / 100.0, 0.01)

    return html.Div(
        [
            html.Label(label, className="fw-semibold small mt-3"),
            dcc.RangeSlider(
                id=slider_id,
                min=float(vmin),
                max=float(vmax),
                step=float(step),
                value=[float(vmin), float(vmax)],
                marks=make_marks(vmin, vmax),
                tooltip={"placement": "bottom", "always_visible": False},
                allowCross=False,
            ),
        ]
    )


def make_dropdown_options(values, max_options=None):
    vals = pd.Series(values).dropna().astype(str).sort_values().unique().tolist()

    if max_options is not None:
        vals = vals[:max_options]

    return [{"label": v, "value": v} for v in vals]


# =========================================================
# UTILIDADES DE MAPA
# =========================================================

def tile_layer(tile_key):
    tile = MAP_TILES.get(tile_key, MAP_TILES["openstreet"])

    return dl.TileLayer(
        url=tile["url"],
        attribution=tile["attribution"],
    )


def filter_data(
    df,
    mag_range,
    depth_range,
    rrup_range,
    tmax_range,
    soil_values,
    event_values,
    station_values,
):
    dff = df.copy()

    if mag_range is not None:
        dff = dff[dff["magnitude"].between(float(mag_range[0]), float(mag_range[1]))]

    if depth_range is not None:
        dff = dff[dff["zhypo_km"].between(float(depth_range[0]), float(depth_range[1]))]

    if rrup_range is not None:
        dff = dff[dff["rrup_km"].between(float(rrup_range[0]), float(rrup_range[1]))]

    if tmax_range is not None and "tmax" in dff.columns:
        dff = dff[dff["tmax"].between(float(tmax_range[0]), float(tmax_range[1]))]

    if soil_values:
        soil_values = [int(v) for v in soil_values]
        dff = dff[dff["soil_class"].isin(soil_values)]

    if event_values:
        event_values = [str(v) for v in event_values]
        dff = dff[dff["event_id"].astype(str).isin(event_values)]

    if station_values:
        station_values = [str(v) for v in station_values]
        dff = dff[dff["station_code"].astype(str).isin(station_values)]

    return dff


def station_popup(row):
    return html.Div(
        [
            html.B("Estación"),
            html.Br(),
            f"Código: {row['station_code']}",
            html.Br(),
            f"Soil_Class: {row['soil_class']}",
            html.Br(),
            f"Registros: {int(row['n_records'])}",
        ]
    )


def event_popup(row):
    return html.Div(
        [
            html.B("Sismo"),
            html.Br(),
            f"EQID_Code: {row['event_id']}",
            html.Br(),
            f"Magnitud: {row['magnitude']:.2f}",
            html.Br(),
            f"Profundidad: {row['zhypo_km']:.1f} km",
            html.Br(),
            f"Registros: {int(row['n_records'])}",
        ]
    )


def build_station_markers(dff):
    if dff.empty:
        return []

    stations = (
        dff.groupby(["station_code", "station_lat", "station_lon"], dropna=False)
        .agg(
            soil_class=("soil_class", "first"),
            n_records=("event_id", "nunique"),
        )
        .reset_index()
    )

    markers = []

    for _, row in stations.iterrows():
        markers.append(
            dl.CircleMarker(
                center=[float(row["station_lat"]), float(row["station_lon"])],
                radius=5,
                color="#1f77b4",
                fill=True,
                fillColor="#1f77b4",
                fillOpacity=0.78,
                weight=1,
                children=[
                    dl.Tooltip(
                        [
                            html.B("Estación"),
                            html.Br(),
                            f"Código: {row['station_code']}",
                            html.Br(),
                            f"Soil_Class: {row['soil_class']}",
                        ]
                    ),
                    dl.Popup(station_popup(row)),
                ],
            )
        )

    return markers


def build_event_markers(dff):
    if dff.empty:
        return []

    events = (
        dff.groupby(["event_id", "event_lat", "event_lon"], dropna=False)
        .agg(
            magnitude=("magnitude", "first"),
            zhypo_km=("zhypo_km", "first"),
            n_records=("station_code", "nunique"),
        )
        .reset_index()
    )

    markers = []

    for _, row in events.iterrows():
        mag = float(row["magnitude"])
        radius = max(6, min(18, 2.2 * mag))

        markers.append(
            dl.CircleMarker(
                center=[float(row["event_lat"]), float(row["event_lon"])],
                radius=radius,
                color="#e74c3c",
                fill=True,
                fillColor="#e74c3c",
                fillOpacity=0.72,
                weight=1,
                children=[
                    dl.Tooltip(
                        [
                            html.B("Sismo"),
                            html.Br(),
                            f"EQID_Code: {row['event_id']}",
                            html.Br(),
                            f"Magnitud: {mag:.2f}",
                            html.Br(),
                            f"Profundidad: {row['zhypo_km']:.1f} km",
                        ]
                    ),
                    dl.Popup(event_popup(row)),
                ],
            )
        )

    return markers


def build_connection_lines(dff, max_lines=1200):
    if dff.empty:
        return []

    sample = dff.head(max_lines).copy()

    lines = []

    for _, row in sample.iterrows():
        lines.append(
            dl.Polyline(
                positions=[
                    [float(row["event_lat"]), float(row["event_lon"])],
                    [float(row["station_lat"]), float(row["station_lon"])],
                ],
                color="#7f8c8d",
                weight=1,
                opacity=0.20,
            )
        )

    return lines


def build_summary(dff):
    if dff.empty:
        return dbc.Alert(
            "No hay registros que cumplan los filtros seleccionados.",
            color="warning",
            className="mb-0",
        )

    n_records = len(dff)
    n_events = dff["event_id"].nunique()
    n_stations = dff["station_code"].nunique()
    mag_min = dff["magnitude"].min()
    mag_max = dff["magnitude"].max()

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Registros", className="text-muted small"),
                            html.H4(f"{n_records:,}", className="mb-0"),
                        ]
                    ),
                    className="soft-card",
                ),
                xs=6,
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Eventos", className="text-muted small"),
                            html.H4(f"{n_events:,}", className="mb-0"),
                        ]
                    ),
                    className="soft-card",
                ),
                xs=6,
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Estaciones", className="text-muted small"),
                            html.H4(f"{n_stations:,}", className="mb-0"),
                        ]
                    ),
                    className="soft-card",
                ),
                xs=6,
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Rango Mw", className="text-muted small"),
                            html.H4(f"{mag_min:.2f} – {mag_max:.2f}", className="mb-0"),
                        ]
                    ),
                    className="soft-card",
                ),
                xs=6,
                md=3,
            ),
        ],
        className="g-3 mb-3",
    )


def legend_component():
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "28px",
                            "height": "2px",
                            "backgroundColor": "#7f8c8d",
                            "marginRight": "8px",
                            "verticalAlign": "middle",
                        }
                    ),
                    html.Span("Conexiones", className="me-4"),
                ],
                style={"display": "inline-block"},
            ),
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "14px",
                            "height": "14px",
                            "borderRadius": "50%",
                            "backgroundColor": "#e74c3c",
                            "marginRight": "8px",
                            "verticalAlign": "middle",
                        }
                    ),
                    html.Span("Sismos", className="me-4"),
                ],
                style={"display": "inline-block"},
            ),
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "10px",
                            "height": "10px",
                            "borderRadius": "50%",
                            "backgroundColor": "#1f77b4",
                            "marginRight": "8px",
                            "verticalAlign": "middle",
                        }
                    ),
                    html.Span("Estaciones"),
                ],
                style={"display": "inline-block"},
            ),
        ],
        className="text-center text-muted small mt-3",
    )


# =========================================================
# LAYOUT
# =========================================================

def layout():
    if dl is None:
        return dbc.Container(
            dbc.Alert(
                [
                    html.H4("Falta instalar dash-leaflet.", className="alert-heading"),
                    html.P(f"Error original: {DL_IMPORT_ERROR}"),
                    html.Code("python -m pip install dash-leaflet"),
                ],
                color="danger",
            ),
            fluid=True,
        )

    try:
        df = load_map_data()
    except Exception as exc:
        return dbc.Container(
            dbc.Alert(
                [
                    html.H4("No se pudo cargar la base del mapa.", className="alert-heading"),
                    html.P(str(exc)),
                ],
                color="danger",
            ),
            fluid=True,
        )

    mag_min, mag_max = get_bounds(df["magnitude"])
    depth_min, depth_max = get_bounds(df["zhypo_km"])
    rrup_min, rrup_max = get_bounds(df["rrup_km"])
    tmax_min, tmax_max = get_bounds(df["tmax"])

    soil_values = sorted([int(x) for x in df["soil_class"].dropna().unique() if int(x) != -1])

    soil_options = [
        {"label": f"Soil_Class {s}", "value": int(s)}
        for s in soil_values
    ]

    event_options = make_dropdown_options(df["event_id"])
    station_options = make_dropdown_options(df["station_code"])

    return dbc.Container(
        [
            html.Div("Visualización espacial", className="section-kicker"),
            html.H1("Mapa interactivo de sismos y estaciones", className="page-title"),
            html.P(
                "Visualización espacial de estaciones, epicentros y conexiones evento-estación, "
                "con filtros dinámicos por variables numéricas y categóricas.",
                className="lead",
            ),

            html.Div(id="mapa-summary", className="mb-3"),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Controles del mapa", className="mb-3"),

                                    html.Label("Fondo cartográfico", className="fw-semibold small"),
                                    dcc.Dropdown(
                                        id="mapa-tile",
                                        options=[
                                            {"label": MAP_TILES[k]["label"], "value": k}
                                            for k in MAP_TILES
                                        ],
                                        value="openstreet",
                                        clearable=False,
                                    ),

                                    html.Label("Capas visibles", className="fw-semibold small mt-4"),
                                    dbc.Checklist(
                                        id="mapa-layers",
                                        options=[
                                            {"label": "Estaciones", "value": "stations"},
                                            {"label": "Sismos", "value": "events"},
                                            {"label": "Conexiones", "value": "connections"},
                                        ],
                                        value=["stations", "events", "connections"],
                                        switch=True,
                                    ),

                                    make_range_slider(
                                        "mapa-mag-range",
                                        "Magnitud",
                                        mag_min,
                                        mag_max,
                                        step=0.01,
                                    ),

                                    make_range_slider(
                                        "mapa-depth-range",
                                        "Profundidad hipocentral (km)",
                                        depth_min,
                                        depth_max,
                                        step=0.1,
                                    ),

                                    make_range_slider(
                                        "mapa-rrup-range",
                                        "Rrup (km)",
                                        rrup_min,
                                        rrup_max,
                                        step=0.5,
                                    ),

                                    make_range_slider(
                                        "mapa-tmax-range",
                                        "Tmax (s)",
                                        tmax_min,
                                        tmax_max,
                                        step=0.1,
                                    ),

                                    html.Label("Soil_Class", className="fw-semibold small mt-3"),
                                    dcc.Dropdown(
                                        id="mapa-soil-filter",
                                        options=soil_options,
                                        value=[],
                                        multi=True,
                                        placeholder="Filtrar por clase de suelo",
                                    ),

                                    html.Label("EQID_Code", className="fw-semibold small mt-3"),
                                    dcc.Dropdown(
                                        id="mapa-event-filter",
                                        options=event_options,
                                        value=[],
                                        multi=True,
                                        placeholder="Filtrar por EQID_Code",
                                    ),

                                    html.Label("Station Code", className="fw-semibold small mt-3"),
                                    dcc.Dropdown(
                                        id="mapa-station-filter",
                                        options=station_options,
                                        value=[],
                                        multi=True,
                                        placeholder="Filtrar por Station Code",
                                    ),

                                    html.Hr(),

                                    html.P(
                                        "Si no seleccionas valores en Soil_Class, EQID_Code o Station Code, "
                                        "se muestran todos los registros disponibles.",
                                        className="text-muted small mb-0",
                                    ),
                                ]
                            ),
                            className="soft-card",
                        ),
                        xs=12,
                        lg=4,
                        xl=3,
                    ),

                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Visualización", className="mb-0")
                                ),
                                dbc.CardBody(
                                    [
                                        dl.Map(
                                            id="mapa-leaflet",
                                            center=[4.6, -74.1],
                                            zoom=5,
                                            children=[
                                                tile_layer("openstreet"),
                                            ],
                                            style={
                                                "height": "780px",
                                                "width": "100%",
                                                "borderRadius": "18px",
                                                "overflow": "hidden",
                                            },
                                        ),
                                        legend_component(),
                                    ]
                                ),
                            ],
                            className="soft-card",
                        ),
                        xs=12,
                        lg=8,
                        xl=9,
                    ),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )


# =========================================================
# CALLBACKS
# =========================================================

def register_callbacks(app):
    @app.callback(
        Output("mapa-leaflet", "children"),
        Output("mapa-summary", "children"),
        Input("mapa-tile", "value"),
        Input("mapa-layers", "value"),
        Input("mapa-mag-range", "value"),
        Input("mapa-depth-range", "value"),
        Input("mapa-rrup-range", "value"),
        Input("mapa-tmax-range", "value"),
        Input("mapa-soil-filter", "value"),
        Input("mapa-event-filter", "value"),
        Input("mapa-station-filter", "value"),
    )
    def update_map(
        tile_key,
        layers,
        mag_range,
        depth_range,
        rrup_range,
        tmax_range,
        soil_values,
        event_values,
        station_values,
    ):
        if dl is None:
            return [], dbc.Alert("dash-leaflet no está instalado.", color="danger")

        layers = layers or []

        try:
            df = load_map_data()

            dff = filter_data(
                df=df,
                mag_range=mag_range,
                depth_range=depth_range,
                rrup_range=rrup_range,
                tmax_range=tmax_range,
                soil_values=soil_values,
                event_values=event_values,
                station_values=station_values,
            )

            children = [tile_layer(tile_key)]

            if "connections" in layers:
                children.append(
                    dl.LayerGroup(
                        build_connection_lines(dff),
                        id="mapa-connections-layer",
                    )
                )

            if "events" in layers:
                children.append(
                    dl.LayerGroup(
                        build_event_markers(dff),
                        id="mapa-events-layer",
                    )
                )

            if "stations" in layers:
                children.append(
                    dl.LayerGroup(
                        build_station_markers(dff),
                        id="mapa-stations-layer",
                    )
                )

            children.append(dl.ScaleControl(position="bottomleft"))

            return children, build_summary(dff)

        except Exception as exc:
            return [tile_layer(tile_key)], dbc.Alert(str(exc), color="danger")