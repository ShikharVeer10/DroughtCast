import os
import cdsapi

os.makedirs("data", exist_ok=True)

client = cdsapi.Client()

print("Downloading ERA5 monthly means dataset from Copernicus CDS...")

client.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            '2m_temperature',
            'mean_sea_level_pressure',
            'total_precipitation'
        ],
        'year': ['2023', '2024', '2025'],
        'month': [
            '01', '02', '03', '04', '05', '06', 
            '07', '08', '09', '10', '11', '12'
        ],
        'time': '00:00',
        'format': 'netcdf',
        'area': [20, 70, 10, 85],  # [North, West, South, East] bounding box
    },
    'data/era5_inputs.nc'
)

print("Success! Downloaded file saved to data/era5_inputs.nc")