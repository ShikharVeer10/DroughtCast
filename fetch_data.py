import os
import requests
import pandas as pd

API_KEY = "Ud9D6HuoXkf3qQ5WLjKFbCyiahHhERVmHS0iFDuF"  
url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

latitude = 15.9129
longitude = 79.7400

parameters = "PRECTOTCORR,T2M,T2MDEW,WS10M,ALLSKY_SFC_SW_DWN"

params = {
    "parameters": parameters,
    "community": "RE",       
    "longitude": longitude,
    "latitude": latitude,
    "start": 1995,           
    "end": 2020,             
    "format": "JSON",
    "api_key": API_KEY
}

print("Fetching data from NASA POWER API")
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    parameter_data = data["properties"]["parameter"]
    df = pd.DataFrame.from_dict(parameter_data)
    df.index.name = "YearMonth"
    os.makedirs("data", exist_ok=True)
    output_path = "data/era5_inputs.csv"
    df.to_csv(output_path)
    
    print(f"Successfully saved climate data to {output_path}!")
else:
    print(f"Error {response.status_code}: {response.text}")