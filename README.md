# Detroit ShotStoppers – RMS/911 Pipeline + DiD Analysis

This repository contains a pipeline to:
1) Pull and clean Detroit RMS Crime Incidents and Police-Serviced 911 Calls
2) Map incidents to CVI geographies (with boundary buffer handling)
3) Run exploratory data analysis on the RMS data
4) Apply fractional assignment for boundary/buffer points
5) Run baseline DiD (Python) and dynamic DiD (R: DIDmultiplegtDYN)

# Pipeline Order

1) `python src/grab_clean_rms.py`
2) `python src/grab_clean_911.py`
3) `python src/cvi_mapping.py`
4) `python src/eda.py`
5) `python src/fractional_assignment.py`
6) `python src/did.py`
7) `Rscript src/run_did_multiplegt_dyn.R`

## Inputs / Outputs

- Inputs are pulled from Detroit’s Open Data portal / ArcGIS endpoints.
- Outputs (CSVs, HTML maps, plots) are written locally and are not committed to git (see `.gitignore`).

### Python environment (example)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

# To initially write requirements.txt
pip install pandas numpy geopandas shapely pyproj requests plotly statsmodels scipy folium
pip freeze > requirements.txt

# To install Python libraries later
pip install -r requirements.txt

# R dependencies
# R packages installed from an R session

```r
install.packages(c("readr", "dplyr"))
install.packages("DIDmultiplegtDYN")

# Run R in powershell such as VS Code

& "C:\Program Files\R\R-4.5.2\bin\R.exe"
