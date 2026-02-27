import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ============================================================
# USER PARAMETERS
# ============================================================
aggregation_hours = 24
winter_years = list(range(1995, 2025)) 
# Select NUTS-3 codes
selected_nuts3 = ["DEF07", "DEF05"]

era5_10m = Path("../winter_1995_2025/era5_10m")
era5_100m = Path("../winter_1995_2025/era5_100m")
hostrada_folder = Path("../winter_1995_2025/hostrada/")
output_folder = Path("prod_data/")
output_folder.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD NUTS-3 SHAPEFILE AND LOAD WIND FARMS AND FILTER BY NUTS-3
# ============================================================
nuts3 = gpd.read_file("map_de/nuts/NUTS250_N3.shp")
nuts3 = nuts3.to_crs(epsg=3034)
nuts3_sel = nuts3[nuts3["NUTS_CODE"].isin(selected_nuts3)]
region_polygon = nuts3_sel.unary_union

# Use the unary_union polygon of selected NUTS-3 regions
file_path = "renewable_power_plants/opendata_wka_ib_gv_vb_sh_202500707.xlsx"

df_wind = pd.read_excel(file_path)
# Create GeoDataFrame (WGS84) and convert to HOSTRADA projection (EPSG:3034)
wind_farms_gdf = gpd.GeoDataFrame(
    df_wind,
    geometry=gpd.points_from_xy(df_wind.OSTWERT, df_wind.NORDWERT),
    crs="EPSG:25832"
).to_crs(epsg=3034)
df_wind_sel = wind_farms_gdf[wind_farms_gdf.geometry.within(region_polygon)]

print(f"Selected {len(df_wind_sel)} wind turbines in chosen NUTS-3")

# Create GeoDataFrame (WGS84) and convert to HOSTRADA projection (EPSG:3034)
wind_farms_gdf = gpd.GeoDataFrame(
    df_wind_sel,
    geometry=gpd.points_from_xy(df_wind_sel.OSTWERT, df_wind_sel.NORDWERT),
    crs="EPSG:25832"
).to_crs(epsg=3034)
print(wind_farms_gdf)
# ============================================================
# 2. SELECT PIXELS CONTAINING WIND FARMS (HOSTRADA GRID)
# ============================================================
ref = list(hostrada_folder.glob("*202301*.nc"))[0]
ds_ref = xr.open_dataset(ref)

X = ds_ref["X"].values
Y = ds_ref["Y"].values
lats = ds_ref["lat"].values
lons = ds_ref["lon"].values

pixel_polygons = []
pixel_indices = []
wind_farms_per_pixel = []  # New list to store wind farm counts

wind = ds_ref["sfcWind"]
mean_wind = wind.mean(dim="time")

pixel_polygons = []
pixel_indices = []
wind_farms_per_pixel = []  # Initialize the list
power_per_pixel = []  # Initialize the list


wind_farms_sindex = wind_farms_gdf.sindex

for i in range(len(Y)-1):
    for j in range(len(X)-1):

        # check wind first (fastest filter)
        w_val = mean_wind.values[i, j]
        if not np.isfinite(w_val):
            continue

        cell = box(X[j], Y[i], X[j+1], Y[i+1])

        # Find wind farms in this cell
        possible_matches_idx = list(wind_farms_sindex.intersection(cell.bounds))
        
        if possible_matches_idx:
            # Filter to get exact matches within the polygon
            exact_matches_idx = []
            for idx in possible_matches_idx:
                if wind_farms_gdf.geometry.iloc[idx].within(cell):
                    exact_matches_idx.append(idx)
            
            count = len(exact_matches_idx)            
            # Only include pixels with at least one wind farm
            if count > 0:
                pixel_wind_farms = wind_farms_gdf.iloc[exact_matches_idx]
                total_power = pixel_wind_farms['LEISTUNG'].sum()
                pixel_polygons.append(cell)
                pixel_indices.append((i, j))
                wind_farms_per_pixel.append(count)
                power_per_pixel.append(total_power)

# Keep ERA5 lat/lon mapping from these pixel centers
pixel_latlon = [(lats[i, j], lons[i, j]) for i, j in pixel_indices]
print(f"Selected pixels with wind farms: {len(pixel_latlon)}")
print(f"Total wind farms counted: {sum(wind_farms_per_pixel)}")

# Create dataframe with coordinates and wind farm counts
pixel_xy_3034 = [(X[j], Y[i]) for i, j in pixel_indices]

from shapely.geometry import Point

# Create dataframe with all information
pixel_data = pd.DataFrame({
    'x': [x for x, y in pixel_xy_3034],
    'y': [y for x, y in pixel_xy_3034],
    'lat': [lat for lat, lon in pixel_latlon],
    'lon': [lon for lat, lon in pixel_latlon],
    'wind_farm_count': wind_farms_per_pixel,
    'total_power': power_per_pixel,
    'i_index': [i for i, j in pixel_indices],
    'j_index': [j for i, j in pixel_indices]
})

# Save to CSV
pixel_data.to_csv("prod_data/pixels_with_wind_farm_counts.csv", index=False)

# Optional: Create a GeoDataFrame with the polygons for visualization
gdf_pixels_with_counts = gpd.GeoDataFrame(
    pixel_data,
    geometry=[Point(x, y) for x, y in pixel_xy_3034],
    crs="EPSG:3034"
)

print(f"\nWind farm count statistics:")
print(f"Min: {min(wind_farms_per_pixel)}")
print(f"Max: {max(wind_farms_per_pixel)}")
print(f"Mean: {sum(wind_farms_per_pixel)/len(wind_farms_per_pixel):.2f}")
print(f"Total pixels with wind farms: {len(wind_farms_per_pixel)}")

# ============================================================
# 3. ERA5 WINTER LOADER
# ============================================================
def load_era5_winter(year):
    """
    Load ERA5 10m and 100m winds for DJF:
    - November + December of <year>
    - January of <year+1>
    """
    def clean_era5(ds):
        if "valid_time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        if "expver" in ds.dims:
            ds = ds.sum("expver")
        for extra in ["forecast_reference_time", "step"]:
            if extra in ds.coords:
                ds = ds.drop_vars(extra)
        for bnd in ["time_bnds", "longitude_bnds", "latitude_bnds"]:
            if bnd in ds.variables:
                ds = ds.drop_vars(bnd)
        return ds

    def open_and_select(filepath, months):
        ds = xr.open_dataset(filepath)
        ds = clean_era5(ds)
        return ds.sel(time=ds["time"].dt.month.isin(months))

    # Build file paths
    f10_this = era5_10m / f"{year}.nc"
    f10_next = era5_10m / f"{year+1}.nc"
    f100_this = era5_100m / f"{year}.nc"
    f100_next = era5_100m / f"{year+1}.nc"

    ds10 = xr.concat([
        open_and_select(f10_this, [12]),
        open_and_select(f10_next, [1, 2])
    ], dim="time").sortby("time")

    ds100 = xr.concat([
        open_and_select(f100_this, [12]),
        open_and_select(f100_next, [1, 2])
    ], dim="time").sortby("time")

    return ds10, ds100
# ============================================================
# 4. PROCESS WINTERS (with hourly-monthly alpha)
# ============================================================
all_daily100 = []
all_daily10 = []
all_alpha_max = []
df_lat_lon = None

# Create storage for plots
hostrada_10m_data = []
era5_10m_data = []
hostrada_100m_data = []
era5_100m_data = []

# Configuration parameter
ALPHA_FREQUENCY = 'hour_of_day_monthly'

print(f"\n Using alpha frequency: {ALPHA_FREQUENCY}")

for year in winter_years:
    print(f"\n=== WINTER {year}–{year+1} ===")

    # ---- ERA5: compute alpha ----
    ds10, ds100 = load_era5_winter(year)
    w10_era = np.sqrt(ds10["u10"]**2 + ds10["v10"]**2)
    w100_era = np.sqrt(ds100["u100"]**2 + ds100["v100"]**2)
    alpha = (np.log(w100_era) - np.log(w10_era)) / (np.log(100) - np.log(10))

    # ---- Map ERA5 grid to pixel coordinates ----
    pixel_lats = [p[0] for p in pixel_latlon]
    pixel_lons = [p[1] for p in pixel_latlon]
    lat_points = xr.DataArray(pixel_lats, dims="points")
    lon_points = xr.DataArray(pixel_lons, dims="points")
    
    era5_10m_sel = w10_era.sel(latitude=lat_points, longitude=lon_points, method="nearest")
    era5_100m_sel = w100_era.sel(latitude=lat_points, longitude=lon_points, method="nearest")
    alpha_sel = alpha.sel(latitude=lat_points, longitude=lon_points, method="nearest")  

    # ---- Collect HOSTRADA monthly files ----
    prefixes = [f"{year}12", f"{year+1}01", f"{year+1}02"]
    host_files = []
    for pref in prefixes:
        # Use a stricter pattern: match only files starting with the prefix
        files = list(hostrada_folder.glob(f"sfcWind_1hr_HOSTRADA-v1-0_BE_gn_{pref}*.nc"))
        host_files.extend(files)
    ds_host = xr.open_mfdataset(host_files, combine="nested", concat_dim="time")
    h10 = ds_host["sfcWind"]

    # ---- Extract only selected pixels ----
    idx_y = xr.DataArray([i for i,j in pixel_indices], dims='points')
    idx_x = xr.DataArray([j for i,j in pixel_indices], dims='points')
    host_pix = h10.isel(Y=idx_y, X=idx_x).values
    Nt = min(host_pix.shape[0], alpha_sel.shape[0])
    host_pix = host_pix[:Nt, :]
    era5_10m_sel = era5_10m_sel.values[:Nt, :]
    era5_100m_sel = era5_100m_sel.values[:Nt, :]
    alpha_sel = alpha_sel.values[:Nt, :]
    print(f" Winter hours retained: {Nt}")

    # ---- ALPHA PROCESSING: Hour-of-day monthly mean ----
    if ALPHA_FREQUENCY == 'hour_of_day_monthly':
        # We have 3 months: Dec (year), Jan (year+1), Feb (year+1)
        winter_months = 3
        hours_per_month = Nt // winter_months
        
        # Check if we have complete months
        if Nt % winter_months != 0:
            print(f"Warning: {Nt} hours not divisible by 3 months. Trimming to {hours_per_month * winter_months} hours")
            Nt = hours_per_month * winter_months
            host_pix = host_pix[:Nt, :]
            era5_10m_sel = era5_10m_sel[:Nt, :]
            era5_100m_sel = era5_100m_sel[:Nt, :]
            alpha_sel = alpha_sel[:Nt, :]
        
        # Create hour-of-day index (0-23) for each hour
        hours_in_day = 24
        hour_of_day_indices = np.tile(np.arange(hours_in_day), Nt // hours_in_day + 1)[:Nt]
        
        # Process each month separately
        alpha_hourly_monthly = []
        
        for month_idx in range(winter_months):
            start_idx = month_idx * hours_per_month
            end_idx = (month_idx + 1) * hours_per_month
            
            # Extract data for this month
            alpha_month = alpha_sel[start_idx:end_idx, :]
            hour_indices_month = hour_of_day_indices[start_idx:end_idx]
            
            # For each hour of day (0-23), compute mean alpha across the month
            alpha_hour_of_day = np.zeros((hours_in_day, alpha_month.shape[1]))
            
            for hour in range(hours_in_day):
                # Get indices where hour_of_day == hour
                hour_mask = (hour_indices_month == hour)
                if np.any(hour_mask):
                    # Compute mean alpha for this hour across all days in the month
                    alpha_hour_of_day[hour, :] = np.nanmean(alpha_month[hour_mask, :], axis=0)
                else:
                    # If no data for this hour (shouldn't happen), use overall mean
                    alpha_hour_of_day[hour, :] = np.nanmean(alpha_month, axis=0)
            
            # Now create alpha array for this month using hour-of-day values
            # For each hour in the month, use the corresponding hour-of-day alpha
            alpha_month_hourly = np.zeros_like(alpha_month)
            
            for i in range(hours_per_month):
                hour = hour_indices_month[i]
                alpha_month_hourly[i, :] = alpha_hour_of_day[hour, :]
            
            alpha_hourly_monthly.append(alpha_month_hourly)
        
        # Combine all months
        alpha_for_extrap = np.vstack(alpha_hourly_monthly)
        
        print(f"   Using hour-of-day monthly alpha (24 values per month)")
        print(f"   Example - Alpha for hour 0 (midnight): {alpha_hour_of_day[0, 0]:.3f}")
        print(f"   Example - Alpha for hour 12 (noon): {alpha_hour_of_day[12, 0]:.3f}")
        
        # Ensure matching dimensions
        N_hours_extrap = min(host_pix.shape[0], alpha_for_extrap.shape[0])
        host_pix = host_pix[:N_hours_extrap, :]
        alpha_for_extrap = alpha_for_extrap[:N_hours_extrap, :]
        era5_10m_sel = era5_10m_sel[:N_hours_extrap, :]
        era5_100m_sel = era5_100m_sel[:N_hours_extrap, :]
        
    elif ALPHA_FREQUENCY == 'hourly':
        # Original hourly method
        alpha_for_extrap = alpha_sel
        print("   Using hourly alpha values")
        
    elif ALPHA_FREQUENCY == 'daily':
        # Daily mean method
        days = Nt // aggregation_hours
        alpha_hourly_days = alpha_sel[:days*aggregation_hours].reshape(days, aggregation_hours, -1)
        with np.errstate(invalid='ignore'):
            alpha_daily_mean = np.nanmean(alpha_hourly_days, axis=1)
        alpha_for_extrap = np.repeat(alpha_daily_mean, aggregation_hours, axis=0)
        
        N_hours_extrap = min(host_pix.shape[0], alpha_for_extrap.shape[0])
        host_pix = host_pix[:N_hours_extrap, :]
        alpha_for_extrap = alpha_for_extrap[:N_hours_extrap, :]
        era5_10m_sel = era5_10m_sel[:N_hours_extrap, :]
        era5_100m_sel = era5_100m_sel[:N_hours_extrap, :]
        
        print(f"   Using daily mean alpha")
        
    elif ALPHA_FREQUENCY == 'monthly':
        # Monthly mean method (same hour for all hours in month)
        winter_months = 3
        hours_per_month = Nt // winter_months
        
        alpha_monthly_means = []
        for month_idx in range(winter_months):
            start_idx = month_idx * hours_per_month
            end_idx = (month_idx + 1) * hours_per_month
            alpha_month = alpha_sel[start_idx:end_idx, :]
            with np.errstate(invalid='ignore'):
                alpha_month_mean = np.nanmean(alpha_month, axis=0, keepdims=True)
            alpha_month_repeated = np.repeat(alpha_month_mean, hours_per_month, axis=0)
            alpha_monthly_means.append(alpha_month_repeated)
        
        alpha_for_extrap = np.vstack(alpha_monthly_means)
        
        N_hours_extrap = min(host_pix.shape[0], alpha_for_extrap.shape[0])
        host_pix = host_pix[:N_hours_extrap, :]
        alpha_for_extrap = alpha_for_extrap[:N_hours_extrap, :]
        era5_10m_sel = era5_10m_sel[:N_hours_extrap, :]
        era5_100m_sel = era5_100m_sel[:N_hours_extrap, :]
        
        print(f"   Using monthly mean alpha")

    # ---- Vertical extrapolation to 100m ----
    w100_est = host_pix * (100 / 10) ** alpha_for_extrap

    # ---- Store hourly data for plotting ----
    hostrada_10m_data.append(host_pix)
    era5_10m_data.append(era5_10m_sel)
    hostrada_100m_data.append(w100_est)
    era5_100m_data.append(era5_100m_sel)

    # ---- Compute daily maxima and corresponding alpha ----
    Nt_actual = host_pix.shape[0]
    days = Nt_actual // aggregation_hours
    
    # Reshape for daily analysis
    host_pix_reshaped = host_pix[:days*aggregation_hours].reshape(days, aggregation_hours, -1)
    w100_est_reshaped = w100_est[:days*aggregation_hours].reshape(days, aggregation_hours, -1)
    alpha_sel_reshaped = alpha_sel[:days*aggregation_hours].reshape(days, aggregation_hours, -1)
    
    # Find indices of daily maxima at 100m
    max_indices = np.argmax(w100_est_reshaped, axis=1)
    
    # Use these indices to extract corresponding alpha values
    days_idx = np.arange(days)[:, np.newaxis]
    pixels_idx = np.arange(w100_est_reshaped.shape[2])[np.newaxis, :]
    
    # Get alpha values at times of daily maxima
    alpha_at_max = alpha_sel_reshaped[days_idx, max_indices, pixels_idx]
    
    # Compute daily maxima values
    host_pix_daily = host_pix_reshaped.max(axis=1)
    w100_daily = w100_est_reshaped.max(axis=1)

    all_daily10.append(host_pix_daily)
    all_daily100.append(w100_daily)
    all_alpha_max.append(alpha_at_max)

    # Print alpha statistics
    print(f"   Alpha stats - Min: {alpha_for_extrap.min():.3f}, "
          f"Mean: {alpha_for_extrap.mean():.3f}, "
          f"Max: {alpha_for_extrap.max():.3f}")

    # ---- store pixel lat/lon once ----
    if df_lat_lon is None:
        df_lat_lon = pd.DataFrame({
            "lat": pixel_lats,
            "lon": pixel_lons
        })

# ============================================================
# 5. SAVE OUTPUT
# ============================================================
if all_daily100:
    np.savetxt(output_folder / "daily_100m_corrected.csv",
               np.concatenate(all_daily100, axis=0), delimiter=",")
    #np.savetxt(output_folder / "daily_10m.csv",
    #           np.concatenate(all_daily10, axis=0), delimiter=",")
    #np.savetxt(output_folder / "alpha_at_100m_maxima.csv",  # New output
    #           np.concatenate(all_alpha_max, axis=0), delimiter=",")
    #df_lat_lon.to_csv(output_folder / "pixel_locations.csv", index=False)
    print("\n✔ Finished successfully!")
else:
    print("\n❌ No data extracted.")

# ============================================================
# 5. CREATE PLOTS FOR EACH PIXEL
# ============================================================
print("\n=== Creating wind speed comparison plots ===")

# Concatenate all winter data
hostrada_10m_all = np.concatenate(hostrada_10m_data, axis=0)
era5_10m_all = np.concatenate(era5_10m_data, axis=0)
hostrada_100m_all = np.concatenate(hostrada_100m_data, axis=0)
era5_100m_all = np.concatenate(era5_100m_data, axis=0)

# remove columns with any NaN
hostrada_10m_all = hostrada_10m_all[:, ~np.isnan(hostrada_10m_all).any(axis=0)]
era5_10m_all = era5_10m_all[:, ~np.isnan(era5_10m_all).any(axis=0)]
hostrada_100m_all = hostrada_100m_all[:, ~np.isnan(hostrada_100m_all).any(axis=0)]
era5_100m_all = era5_100m_all[:, ~np.isnan(era5_100m_all).any(axis=0)]


# Create output directory for plots
plot_dir = Path("wind_speed_comparisons")
plot_dir.mkdir(exist_ok=True)

# Plot for each pixel
num_pixels = hostrada_10m_all.shape[1]

for pixel_idx in range(num_pixels):
    print(f"  Plotting pixel {pixel_idx+1}/{num_pixels}")
    
    # Get data for this pixel
    host_10m_pixel = hostrada_10m_all[:, pixel_idx]
    era5_10m_pixel = era5_10m_all[:, pixel_idx]
    host_100m_pixel = hostrada_100m_all[:, pixel_idx]
    era5_100m_pixel = era5_100m_all[:, pixel_idx]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Wind Speed Comparison - Pixel {pixel_idx+1}\n'
                 f'Lat: {pixel_lats[pixel_idx]:.3f}°, Lon: {pixel_lons[pixel_idx]:.3f}°',
                 fontsize=14, fontweight='bold')
    
    # 1. Time series at 10m
    time_idx = np.arange(len(host_10m_pixel))
    axes[0, 0].plot(time_idx, host_10m_pixel, 'b-', alpha=0.7, label='HOSTRADA 10m', linewidth=0.5)
    axes[0, 0].plot(time_idx, era5_10m_pixel, 'r-', alpha=0.7, label='ERA5 10m', linewidth=0.5)
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Wind Speed (m/s)')
    axes[0, 0].set_title('Time Series at 10m')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter plot at 10m
    valid_mask_10m = (host_10m_pixel > 0) & (era5_10m_pixel > 0)
    axes[0, 1].scatter(host_10m_pixel[valid_mask_10m], era5_10m_pixel[valid_mask_10m], 
                      alpha=0.5, s=1, c='blue')
    # Add 1:1 line
    max_val = max(host_10m_pixel.max(), era5_10m_pixel.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
    
    # Calculate statistics
    corr_10m = r2_score(host_10m_pixel[valid_mask_10m], era5_10m_pixel[valid_mask_10m])
    bias_10m = np.mean(host_10m_pixel[valid_mask_10m] - era5_10m_pixel[valid_mask_10m])
    
    axes[0, 1].set_xlabel('HOSTRADA 10m (m/s)')
    axes[0, 1].set_ylabel('ERA5 10m (m/s)')
    axes[0, 1].set_title(f'Scatter Plot at 10m\n$R^2$: {corr_10m:.3f}, Bias: {bias_10m:.3f} m/s')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # 3. Time series at 100m
    axes[1, 0].plot(time_idx, host_100m_pixel, 'b-', alpha=0.7, label='HOSTRADA 100m', linewidth=0.5)
    axes[1, 0].plot(time_idx, era5_100m_pixel, 'r-', alpha=0.7, label='ERA5 100m', linewidth=0.5)
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Wind Speed (m/s)')
    axes[1, 0].set_title('Time Series at 100m')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter plot at 100m
    valid_mask_100m = (host_100m_pixel > 0) & (era5_100m_pixel > 0)
    axes[1, 1].scatter(host_100m_pixel[valid_mask_100m], era5_100m_pixel[valid_mask_100m], 
                      alpha=0.5, s=1, c='green')
    # Add 1:1 line
    max_val = max(host_100m_pixel.max(), era5_100m_pixel.max())
    axes[1, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
    
    # Calculate statistics
    corr_100m = r2_score(host_100m_pixel[valid_mask_100m], era5_100m_pixel[valid_mask_100m])
    bias_100m = np.mean(host_100m_pixel[valid_mask_100m] - era5_100m_pixel[valid_mask_100m])
    
    axes[1, 1].set_xlabel('HOSTRADA 100m (m/s)')
    axes[1, 1].set_ylabel('ERA5 100m (m/s)')
    axes[1, 1].set_title(f'Scatter Plot at 100m\n$R^2$: {corr_100m:.3f}, Bias: {bias_10m:.3f} m/s')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = plot_dir / f'wind_comparison_pixel_{pixel_idx+1:03d}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"\n✅ Plots saved in: {plot_dir}")
## Concatenate all winter data
hostrada_10m_all = np.concatenate(hostrada_10m_data, axis=0)
era5_10m_all = np.concatenate(era5_10m_data, axis=0)
hostrada_100m_all = np.concatenate(hostrada_100m_data, axis=0)
era5_100m_all = np.concatenate(era5_100m_data, axis=0)

# remove columns with any NaN
hostrada_10m_all = hostrada_10m_all[:, ~np.isnan(hostrada_10m_all).any(axis=0)]
era5_10m_all = era5_10m_all[:, ~np.isnan(era5_10m_all).any(axis=0)]
hostrada_100m_all = hostrada_100m_all[:, ~np.isnan(hostrada_100m_all).any(axis=0)]
era5_100m_all = era5_100m_all[:, ~np.isnan(era5_100m_all).any(axis=0)]

# Compute R² per pixel
num_pixels = hostrada_10m_all.shape[1]

r2_10m = []
r2_100m = []

for pixel_idx in range(num_pixels):
    host_10m_pixel = hostrada_10m_all[:, pixel_idx]
    era5_10m_pixel = era5_10m_all[:, pixel_idx]
    host_100m_pixel = hostrada_100m_all[:, pixel_idx]
    era5_100m_pixel = era5_100m_all[:, pixel_idx]

    # Only consider positive values
    mask_10m = (host_10m_pixel > 0) & (era5_10m_pixel > 0)
    mask_100m = (host_100m_pixel > 0) & (era5_100m_pixel > 0)

    r2_10m.append(r2_score(host_10m_pixel[mask_10m], era5_10m_pixel[mask_10m]))
    r2_100m.append(r2_score(host_100m_pixel[mask_100m], era5_100m_pixel[mask_100m]))

# -------------------------
# Boxplot
# -------------------------
plt.figure(figsize=(8, 6))
plt.boxplot([r2_10m, r2_100m],
            patch_artist=True,
            showfliers=False,   # hide outliers
            boxprops=dict(facecolor='skyblue', color='blue'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'))

plt.xticks([1, 2], ['ERA5 vs Hostrada 10m', 'ERA5 vs Hostrada 100m'])
plt.ylabel('R² per pixel')
plt.title('Distribution of R² Across Pixels (10m vs 100m)')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
# -------------------------
# Save figure
# -------------------------
plot_dir = Path("wind_speed_comparisons")
plot_dir.mkdir(exist_ok=True)
save_path = plot_dir / "R2_boxplot_10m_100m.pdf"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
#plt.show()
