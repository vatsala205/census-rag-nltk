import pandas as pd
import json
import numpy as np

def analyze_census_data(file_path):
    """Analyze census data and provide insights"""

    # Load the data
    df = pd.read_csv(file_path)  # CSV file with comma separator
    df.columns = df.columns.str.strip()

    print("=== CENSUS DATA ANALYSIS ===\n")

    # Basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    # Filter out summary rows (INDIA @& and INDIA $)
    detailed_df = df[~df['Name'].str.contains('INDIA', na=False)]

    print(f"Detailed records (excluding India totals): {len(detailed_df)}")

    # Population analysis
    print("\n=== POPULATION ANALYSIS ===")
    if 'Population (Persons)' in df.columns:
        total_pop = detailed_df['Population (Persons)'].sum()
        print(f"Total population in detailed records: {total_pop:,}")

        # Top 10 most populated areas
        top_populated = detailed_df.nlargest(10, 'Population (Persons)')
        print("\nTop 10 most populated areas:")
        for i, row in top_populated.iterrows():
            print(f"{row['Name']}: {row['Population (Persons)']:,}")

    # Area analysis
    print("\n=== AREA ANALYSIS ===")
    if 'Area (sq km)' in df.columns:
        total_area = detailed_df['Area (sq km)'].sum()
        print(f"Total area covered: {total_area:,} sq km")

        # Top 10 largest areas
        top_areas = detailed_df.nlargest(10, 'Area (sq km)')
        print("\nTop 10 largest areas:")
        for i, row in top_areas.iterrows():
            print(f"{row['Name']}: {row['Area (sq km)']:,} sq km")

    # Density analysis
    print("\n=== DENSITY ANALYSIS ===")
    if 'Population Density' in df.columns:
        avg_density = detailed_df['Population Density'].mean()
        print(f"Average population density: {avg_density:.2f} people per sq km")

        # Most dense areas
        top_dense = detailed_df.nlargest(10, 'Population Density')
        print("\nTop 10 most densely populated areas:")
        for i, row in top_dense.iterrows():
            print(f"{row['Name']}: {row['Population Density']:,} people per sq km")

    # Rural-Urban analysis
    print("\n=== RURAL-URBAN ANALYSIS ===")
    rural_rows = detailed_df[detailed_df['Total/Rural/Urban'] == 'Rural']
    urban_rows = detailed_df[detailed_df['Total/Rural/Urban'] == 'Urban']

    if len(rural_rows) > 0 and len(urban_rows) > 0:
        rural_pop = rural_rows['Population (Persons)'].sum()
        urban_pop = urban_rows['Population (Persons)'].sum()
        print(f"Rural population: {rural_pop:,}")
        print(f"Urban population: {urban_pop:,}")
        print(f"Rural-Urban ratio: {rural_pop / urban_pop:.2f}")

    # Generate sample queries for the chatbot
    print("\n=== SAMPLE CHATBOT QUERIES ===")
    sample_locations = detailed_df['Name'].head(5).tolist()

    sample_queries = []
    for location in sample_locations:
        sample_queries.extend([
            f"What is the population of {location}?",
            f"How big is {location}?",
            f"How many households in {location}?",
            f"Tell me about villages in {location}"
        ])

    print("You can try these sample queries with your chatbot:")
    for i, query in enumerate(sample_queries[:10], 1):
        print(f"{i}. {query}")

    # Save unique locations for reference
    unique_locations = detailed_df['Name'].unique().tolist()
    with open('available_locations.json', 'w', encoding='utf-8') as f:
        json.dump(unique_locations, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(unique_locations)} unique locations to 'available_locations.json'")

    return df, detailed_df


def create_location_lookup(df):
    """Create a lookup dictionary for quick location searches"""

    lookup = {}

    for _, row in df.iterrows():
        name = row['Name']
        if name and not pd.isna(name) and 'INDIA' not in name:
            # Create various lookup keys
            keys = [
                name.lower(),
                name.lower().replace(' ', ''),
                ' '.join(name.lower().split())
            ]

            # Add each word as a key
            words = name.lower().split()
            keys.extend(words)

            for key in keys:
                if key not in lookup:
                    lookup[key] = []
                lookup[key].append(name)

    # Remove duplicates
    for key in lookup:
        lookup[key] = list(set(lookup[key]))

    with open('location_lookup.json', 'w', encoding='utf-8') as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)

    print(f"Created location lookup with {len(lookup)} search keys")

    return lookup


if __name__ == "__main__":
    # Analyze the data
    file_path = 'feature_engineered.csv'  # Your actual CSV file
    try:
        df, detailed_df = analyze_census_data(file_path)
        create_location_lookup(detailed_df)
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Please update the file path.")
    except Exception as e:
        print(f"Error analyzing data: {e}")