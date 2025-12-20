import pandas as pd
from datetime import datetime

def get_hurricane_data(storm_name, storm_year, file_path='hurdat2.txt'):
    """
    Parses the HURDAT2 text file and extracts the data for a specific
    hurricane by name and year.
    
    Returns a pandas DataFrame with the storm's track and intensity data.
    """
    print(f"Parsing '{file_path}' for {storm_name} ({storm_year})...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    storm_data = []
    
    # Enumerate to get both index and line
    for i, line in enumerate(lines):
        parts = line.strip().split(',')
        # Check for a header line
        if len(parts) > 2 and parts[0].startswith('AL'):
            header_id = parts[0].strip()
            header_name = parts[1].strip()
            num_records = int(parts[2].strip())
            
            try:
                header_year = int(header_id[4:8])
            except (ValueError, IndexError):
                continue

            # Check for a match (case-insensitive for the name)
            if header_name == storm_name.upper() and header_year == storm_year:
                print(f"  -> Found data for {storm_name} ({storm_year}). Reading {num_records} records.")
                
                # If we find the storm, loop through its specific data lines and then break
                for data_line_index in range(i + 1, i + 1 + num_records):
                    data_parts = lines[data_line_index].strip().split(',')
                    try:
                        date_str = data_parts[0]
                        time_str = data_parts[1].strip()
                        dt_obj = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")
                        
                        record_id = data_parts[2].strip()
                        status = data_parts[3].strip()
                        lat_str = data_parts[4].strip()
                        lon_str = data_parts[5].strip()
                        max_wind_kts = int(data_parts[6].strip())
                        min_pressure_mb = int(data_parts[7].strip())

                        lat = float(lat_str[:-1]) * (1 if lat_str[-1] == 'N' else -1)
                        lon = float(lon_str[:-1]) * (1 if lon_str[-1] == 'E' else -1)

                        storm_data.append([dt_obj, record_id, status, lat, lon, max_wind_kts, min_pressure_mb])
                    except (ValueError, IndexError):
                        continue
                
                # We've collected all data for our storm, so we can exit the main loop
                break

    if not storm_data:
        print(f"  -> WARNING: No data found for {storm_name} ({storm_year}).")
        return None

    columns = ['datetime', 'record_id', 'status', 'latitude', 'longitude', 'max_wind_kts', 'min_pressure_mb']
    df = pd.DataFrame(storm_data, columns=columns)
    print(f"  -> Successfully created DataFrame with {len(df)} records.")
    
    return df

if __name__ == '__main__':
    # Test cases
    print("--- Testing HURDAT2 Parser ---")
    harvey_df = get_hurricane_data('HARVEY', 2017)
    if harvey_df is not None:
        print("\nFirst 5 records for Hurricane Harvey (2017):")
        print(harvey_df.head())
    
    print("\n--- Testing with another storm ---")
    irma_df = get_hurricane_data('IRMA', 2017)
    if irma_df is not None:
        print("\nFirst 5 records for Hurricane Irma (2017):")
        print(irma_df.head())