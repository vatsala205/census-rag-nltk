import pandas as pd
import json

def prepare_census_data(csv_file_path, output_json_path):
    """
    Prepare census data with separate Total, Rural, and Urban entries for each location
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Dictionary to store processed data
        location_data = {}

        # Process each row
        for index, row in df.iterrows():
            try:
                # Extract basic information
                name = str(row['Name']).strip()
                category = str(row['Total/Rural/Urban']).strip()

                # Skip if name is invalid
                if name in ['nan', 'NaN', ''] or pd.isna(name):
                    continue

                # Clean and extract numeric data with proper error handling
                def safe_convert(value, default=0):
                    try:
                        if pd.isna(value) or value in ['', 'nan', 'NaN']:
                            return default
                        return float(str(value).replace(',', '').strip())
                    except:
                        return default

                # Extract all demographic data
                villages = safe_convert(row.get('Number of Villages (Inhabited)', 0))
                uninhabited = safe_convert(row.get('Number of Villages (Uninhabited)', 0))
                towns = safe_convert(row.get('Number of Towns', 0))
                households = safe_convert(row.get('Number of Households', 0))
                population = safe_convert(row.get('Population (Persons)', 0))
                males = safe_convert(row.get('Population (Males)', 0))
                females = safe_convert(row.get('Population (Females)', 0))
                area = safe_convert(row.get('Area (sq km)', 0))
                density = safe_convert(row.get('Population Density', 0))
                ratio = safe_convert(row.get('Rural-Urban Ratio', 0))

                # Create data structure for this entry
                entry_data = {
                    'villages': int(villages),
                    'uninhabited': int(uninhabited),
                    'towns': int(towns),
                    'households': int(households),
                    'population': int(population),
                    'males': int(males),
                    'females': int(females),
                    'area': round(area, 2),
                    'density': int(density),
                    'ratio': round(ratio, 6)
                }

                # Create location key with category
                location_key = f"{name} {category}"

                # Store the data
                location_data[location_key] = entry_data

                # Print progress for debugging
                if index % 1000 == 0:
                    print(f"Processed {index} rows...")

            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue

        # Save to JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(location_data, f, indent=2, ensure_ascii=False)

        print(f"\nData processing complete!")
        print(f"Total locations processed: {len(location_data)}")
        print(f"Data saved to: {output_json_path}")

        # Print sample entries for verification
        print("\nSample entries:")
        sample_count = 0
        for key, value in location_data.items():
            if sample_count < 5:
                print(f"{key}: Population = {value['population']:,}")
                sample_count += 1
            else:
                break

        # Print statistics
        total_entries = len([k for k in location_data.keys() if 'Total' in k])
        rural_entries = len([k for k in location_data.keys() if 'Rural' in k])
        urban_entries = len([k for k in location_data.keys() if 'Urban' in k])

        print(f"\nData breakdown:")
        print(f"Total entries: {total_entries}")
        print(f"Rural entries: {rural_entries}")
        print(f"Urban entries: {urban_entries}")

        return True

    except Exception as e:
        print(f"Error in prepare_census_data: {e}")
        return False


def verify_data_structure(json_file_path):
    """
    Verify the data structure and show examples
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("Data structure verification:")
        print(f"Total entries: {len(data)}")

        # Find examples of Total, Rural, Urban for the same location
        locations = {}
        for key in data.keys():
            if ' Total' in key:
                base_name = key.replace(' Total', '')
                if base_name not in locations:
                    locations[base_name] = {}
                locations[base_name]['Total'] = key
            elif ' Rural' in key:
                base_name = key.replace(' Rural', '')
                if base_name not in locations:
                    locations[base_name] = {}
                locations[base_name]['Rural'] = key
            elif ' Urban' in key:
                base_name = key.replace(' Urban', '')
                if base_name not in locations:
                    locations[base_name] = {}
                locations[base_name]['Urban'] = key

        # Show examples
        print("\nExample data structure:")
        count = 0
        for base_name, entries in locations.items():
            if count < 3 and len(entries) >= 2:  # Show locations with at least 2 categories
                print(f"\nLocation: {base_name}")
                for category, full_key in entries.items():
                    pop = data[full_key]['population']
                    print(f"  {category}: {pop:,} people")
                count += 1

        return True

    except Exception as e:
        print(f"Error verifying data: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Update these paths according to your file locations
    csv_file_path = "feature_engineered_cleaned.csv"  # Your CSV file path
    output_json_path = "location_data.json"  # Output JSON file path

    print("Starting census data preparation...")

    # Prepare the data
    success = prepare_census_data(csv_file_path, output_json_path)

    if success:
        print("\nVerifying data structure...")
        verify_data_structure(output_json_path)
    else:
        print("Data preparation failed!")