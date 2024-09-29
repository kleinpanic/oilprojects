import requests

# Your EIA API key
API_KEY = 'cxKicy0icuGitsUWHFoL92cHgsRP2i9eYcx8fGPw'
BASE_URL = 'https://api.eia.gov/v2/'

def fetch_crude_oil_imports(start_date, end_date, frequency='monthly'):
    url = f"{BASE_URL}crude-oil-imports/data/"
    
    params = {
        'api_key': API_KEY,
        'frequency': frequency,
        'data[0]': 'quantity',  # The required data type
        'start': start_date,
        'end': end_date,
        'sort[0][column]': 'period',  # Sorting by period
        'sort[0][direction]': 'desc',  # Sorting in descending order
        'offset': 0,
        'length': 5000  # Maximum number of records to fetch
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")

if __name__ == "__main__":
    start_date = "2024-01"  # Adjust the start date as needed
    end_date = "2024-06"    # Adjust the end date as needed
    
    # Fetch crude oil imports data
    crude_oil_imports_data = fetch_crude_oil_imports(start_date, end_date)
    print("Crude Oil Imports Data:")
    print(crude_oil_imports_data)
