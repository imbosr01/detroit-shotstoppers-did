# Detroit ShotStoppers – RMS/911 Pipeline + DiD Analysis

This repository contains a full data pipeline and causal analysis of Detroit’s ShotStoppers Community Violence Intervention (CVI) program using:

- RMS crime incident data
- Police-serviced 911 calls
- Geospatial CVI boundaries

The project includes **two analytical pipelines**:

1. A **baseline citywide DID pipeline**
2. A **matched-pairs causal inference pipeline**

# Data Sources

- RMS Crime Incidents(https://data.detroitmi.gov/datasets/8e532daeec1149879bd5e67fdd9c8be0_0/explore?location=42.348151%2C-83.095882%2C10)
- Police-Serviced 911 Calls(https://data.detroitmi.gov/datasets/5868975fa1e7444cae8ca5240fc77c5b_0/explore)
- CVI Shapefiles(https://data.detroitmi.gov/datasets/46b6151c37684c1ea01bdde0c3e72d13_0/explore?location=42.389585%2C-83.088929%2C11)

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
```

# Baseline Citywide DID Pipeline Order

1) `python src/grab_clean_rms.py`
2) `python src/grab_clean_911.py`
3) `python src/cvi_mapping.py`
4) `python src/eda.py`
5) `python src/fractional_assignment.py`
6) `python src/did.py`
7) `Rscript src/run_did_multiplegt_dyn.R`

# Matched-Pairs Causal Inference Pipeline

1) `python src/grab_clean_rms.py`
2) `python src/grab_clean_911.py`
3) `python src/cvi_mapping.py`
4) `python src/cvi_demographics.py`
5) `python src/build_candidate_control_zones.py`
6) `python src/match_cvis_to_controls.py`
7) `python src/matched_eda.py`
8) `python src/matched_fractional_assignment.py`
9) `python src/matched_did.py`
	- DID with all matched pairs and without Live in Peace & Team Pursuit included
10) `python src/matched_event_study.py`
