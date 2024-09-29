import requests

# Your EIA API key
API_KEY = 'cxKicy0icuGitsUWHFoL92cHgsRP2i9eYcx8fGPw'
BASE_URL = 'https://api.eia.gov/v2/'

def fetch_crude_oil_imports(start_date, end_date, frequency='monthly', origin_id=None, destination_id=None, grade_id=None, sort_by='period', sort_direction='desc'):
    url = f"{BASE_URL}crude-oil-imports/data/"
    
    # Define basic parameters for the API request
    params = {
        'api_key': API_KEY,
        'frequency': frequency,
        'data[0]': 'quantity',  # The required data type (quantity)
        'start': start_date,
        'end': end_date,
        'sort[0][column]': sort_by,  # Sort by the selected field (e.g., period)
        'sort[0][direction]': sort_direction,  # Sort direction: ascending (asc) or descending (desc)
        'offset': 0,
        'length': 5000  # Fetch up to 5000 records
    }

    # Add facets if they are provided (filter by origin, destination, or grade)
    if origin_id:
        params[f'facets[originid][]'] = origin_id  # E.g., origin_id = 'CAN' for Canada
    if destination_id:
        params[f'facets[destinationid][]'] = destination_id  # E.g., destination_id = 'USA' for United States
    if grade_id:
        params[f'facets[gradeid][]'] = grade_id  # E.g., grade_id = 'WTR' for West Texas Intermediate

    # Send the request to the API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return the JSON data
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")

# Function to fetch crude oil import data based on various facets and sorting options
if __name__ == "__main__":
    # Set the desired start and end dates for the data
    start_date = "2024-01"
    end_date = "2024-06"
    
    # Set facet values (can be modified based on your needs)
    origin_id = "CAN"  # Example: Crude oil from Canada
    destination_id = "USA"  # Example: Crude oil imported into the USA
    grade_id = None  # Example: You can specify a grade like "WTR" (West Texas Intermediate) if desired
    
    # Set sorting options
    sort_by = "quantity"  # Sort by quantity of oil
    sort_direction = "desc"  # Sort in descending order

    try:
        # Fetch crude oil imports with the specified facets and sorting
        crude_oil_imports_data = fetch_crude_oil_imports(start_date, end_date, 
                                                         origin_id=origin_id, 
                                                         destination_id=destination_id, 
                                                         grade_id=grade_id, 
                                                         sort_by=sort_by, 
                                                         sort_direction=sort_direction)
        
        # Print the retrieved data
        print("Crude Oil Imports Data:")
        print(crude_oil_imports_data)

    except Exception as e:
        print(f"Error fetching oil data: {e}")
