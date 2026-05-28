# Dashboard de aceleraciones espectrales

Proyecto Dash modular para análisis y predicción de aceleraciones espectrales mediante aprendizaje residual GRU.

## Estructura

```text
app.py
model/
  train_model.py
  model.pkl              # generado después de entrenar
tabs/
  introduccion.py
  problema.py
  objetivos.py
  resultados.py
  prediccion.py
  limitaciones.py
  conclusiones.py
assets/
  styles.css
requirements.txt
```

## Datos esperados

Coloca estos archivos en la raíz del proyecto:

- `Resids_for_Eliasib.xlsx`
- `CopiaDataBaseSGC2.xlsx`

La base principal debe estar en formato largo con columnas como:

- `Record Sequence Number`
- `EQID_Code`
- `Station Code`
- `Hypocenter Depth (km)`
- `Magnitude`
- `Rrup_OpenQuake`
- `Cat` o `Soil_Class`
- `Tcorner`, `Tmax` o `T_max`
- `Period`
- `Sa`
- `Total`
- `Rvolc [km]`

## Ejecución

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python model/train_model.py --models nosam,nosam_elevation --epochs 60
python app.py
```

Para pruebas rápidas sin LOGO:

```bash
python model/train_model.py --models nosam --epochs 5 --skip-logo --verbose 1
```

Para ASK14:

```bash
pip install openquake.engine
python model/train_model.py --models ask14 --epochs 60
```