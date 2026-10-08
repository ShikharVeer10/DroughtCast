import xarray as xr

files = ["era5_inputs.nc", "spei03.nc", "spei06.nc", "spei12.nc"]
for file_path in files:
    try:
        ds = xr.open_dataset(file_path, engine='netcdf4')
        print(f"Dataset: {file_path}")
        print("Variables")
        print(list(ds.data_vars))
        print("\nDimensions")
        print(dict(ds.dims))
        print("\n Coordinates")
        print(list(ds.coords))
        ds.close()
    except Exception as e:
        print(f"Could not open {file_path}: {e}")