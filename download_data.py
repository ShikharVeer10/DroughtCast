import os
import sys
import numpy as np

os.makedirs("data", exist_ok=True)
ERA5_VARIABLES = [
    '10m_u_component_of_wind',       
    '10m_v_component_of_wind',       
    '2m_temperature',               
    'mean_sea_level_pressure',       
    'total_precipitation',          
    'evaporation_from_bare_soil',    
    'surface_thermal_radiation_downwards',  
    'surface_solar_radiation_downwards',    
    'volumetric_soil_water_layer_1', 
    'volumetric_soil_water_layer_2', 
    'runoff',                        
    'sea_surface_temperature',     
]
ERA5_SHORT_NAMES = [
    'u10', 'v10', 't2m', 'msl', 'tp', 'evabs',
    'strd', 'ssrd', 'swvl1', 'swvl2', 'ro', 'sst'
]


def generate_synthetic_data(num_months: int = 60):
    try:
        import xarray as xr
    except ImportError:
        print("ERROR: xarray is required. Install it with:  pip install xarray netcdf4")
        sys.exit(1)

    print(f"Generating synthetic data ({num_months} monthly time steps)...")

    rng = np.random.default_rng(seed=42)
    time_coord = np.arange(num_months)
    data_vars = {}
    for var_name in ERA5_SHORT_NAMES:
        noise = rng.normal(0, 1, size=num_months).cumsum()
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)
        data_vars[var_name] = ("time", noise.astype(np.float32))

    ds_inputs = xr.Dataset(data_vars, coords={"time": time_coord})
    ds_inputs.to_netcdf("data/era5_inputs.nc")
    print("  -> Saved data/era5_inputs.nc")

    spei_3  = rng.normal(0, 1, size=num_months).astype(np.float32)
    spei_6  = rng.normal(0, 1, size=num_months).astype(np.float32)
    spei_12 = rng.normal(0, 1, size=num_months).astype(np.float32)

    ds_targets = xr.Dataset(
        {
            "spei_3":  ("time", spei_3),
            "spei_6":  ("time", spei_6),
            "spei_12": ("time", spei_12),
        },
        coords={"time": time_coord},
    )
    ds_targets.to_netcdf("data/spei_targets.nc")
    print("  -> Saved data/spei_targets.nc")
    print("Synthetic data generation complete!")


def download_from_cds():
    """Download real ERA5 monthly means from Copernicus CDS."""
    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi is required for real data download.")
        print("  Install it with:  pip install cdsapi")
        print("  Then configure ~/.cdsapirc with your CDS API key.")
        print("\n  Alternatively, run with --synthetic for local testing.")
        sys.exit(1)

    client = cdsapi.Client()

    print("Downloading ERA5 monthly means dataset from Copernicus CDS...")
    print(f"  Variables: {len(ERA5_VARIABLES)}")

    client.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': ERA5_VARIABLES,
            'year': ['2020', '2021', '2022', '2023', '2024', '2025'],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12'
            ],
            'time': '00:00',
            'format': 'netcdf',
            'area': [20, 70, 10, 85],  # [North, West, South, East]
        },
        'data/era5_inputs.nc'
    )
    print("Success! Downloaded file saved to data/era5_inputs.nc")

    print("\nNOTE: SPEI target data (data/spei_targets.nc) must be prepared")
    print("separately from a SPEI dataset or computed from precipitation &")
    print("temperature data. Run with --synthetic for a quick test dataset.")


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        generate_synthetic_data()
    else:
        download_from_cds()