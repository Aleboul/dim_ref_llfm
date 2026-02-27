# Data Acquisition Guide

This directory contains the scripts and instructions required to download and preprocess all meteorological datasets used to construct hub-height (100 m) wind speed time series.

The workflow combines two complementary sources:

- ERA5 reanalysis data from the European Centre for Medium-Range Weather Forecasts (ECMWF)
- HOSTRADA high-resolution gridded wind speeds from the Deutscher Wetterdienst (DWD)

ERA5 provides wind components at 10 m and 100 m above ground.
HOSTRADA provides hourly 10 m wind speed on a 1 km × 1 km grid.
HOSTRADA winds are later extrapolated to 100 m using a power-law profile.

---------------------------------------------------------------------

1. ERA5 Data Download

Install CDS API:

pip install cdsapi

Create configuration file:

~/.cdsapirc

Add your personal CDS API key from the Copernicus Climate Data Store.

---------------------------------------------------------------------

Download script

Use the provided script:

API.py

Run:

```console
python API.py
```
The script downloads ERA5 data as NetCDF files.

---------------------------------------------------------------------

Required ERA5 variables

Request:

- 10m_u_component_of_wind
- 10m_v_component_of_wind
- 100m_u_component_of_wind
- 100m_v_component_of_wind

Use hourly resolution.

---------------------------------------------------------------------

Months to download

| Year       | Months                     |
|------------|----------------------------|
| 1995       | December only              |
| 1996–2024  | December, January, February|
| 2025       | January, February only     |

This covers winters continuously from December 1995 through February 2025.

---------------------------------------------------------------------

Example month selection in API.py

December only:
```Python
months = ["12"]
```

January–February only:
```Python
months = ["01", "02"]
```

Full winter:

```Python
months = ["12", "01", "02"]
```
---------------------------------------------------------------------

2. HOSTRADA Data Download

HOSTRADA provides:

- 10 m wind speed
- hourly resolution
- 1 km × 1 km grid
- Germany-wide coverage

Download manually from:

https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/wind_speed/

---------------------------------------------------------------------

Required period

Download ALL winter months:

- December 1995
- Every December–January–February for intermediate years
- January–February 2025

In short:
All winter months between December 1995 and February 2025.