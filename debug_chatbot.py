import json
import pandas as pd


def debug_location_data():
    """Debug the location data to understand what's available"""

    print("=== DEBUGGING LOCATION DATA ===\n")

    # Load location data
    try:
        with open('location_data.json', 'r', encoding='utf-8') as f:
            location_data = json.load(f)
        print(f"✓ Loaded {len(location_data)} locations from location_data.json")
    except FileNotFoundError:
        print("❌ location_data.json not found. Run prepare_census_data.py first.")
        return

    # Show sample locations
    print("\n=== SAMPLE LOCATIONS ===")
    sample_locations = list(location_data.keys())[:10]
    for i, loc in enumerate(sample_locations, 1):
        data = location_data[loc]
        print(f"{i}. {loc}: Pop={data['population']:,.0f}, Area={data['area']:,.0f}")

    # Search for specific locations mentioned in your test
    search_terms = ['pulwama', 'kangan', 'punch', 'poonch']
    print(f"\n=== SEARCHING FOR SPECIFIC LOCATIONS ===")

    for term in search_terms:
        print(f"\nSearching for '{term}':")
        matches = []
        for loc in location_data.keys():
            if term.lower() in loc.lower():
                matches.append(loc)

        if matches:
            print(f"  Found {len(matches)} matches:")
            for match in matches[:5]:  # Show first 5 matches
                data = location_data[match]
                print(f"    - {match}: Pop={data['population']:,.0f}")
        else:
            # Try partial matching
            partial_matches = []
            for loc in location_data.keys():
                if any(word in loc.lower() for word in term.lower().split()):
                    partial_matches.append(loc)

            if partial_matches:
                print(f"  No exact matches, but found {len(partial_matches)} partial matches:")
                for match in partial_matches[:3]:
                    print(f"    - {match}")
            else:
                print(f"  No matches found for '{term}'")

    # Analyze data quality
    print(f"\n=== DATA QUALITY ANALYSIS ===")
    zero_population = sum(1 for data in location_data.values() if data['population'] == 0)
    total_locations = len(location_data)

    print(f"Total locations: {total_locations}")
    print(f"Locations with zero population: {zero_population}")
    print(f"Locations with valid population: {total_locations - zero_population}")

    # Show population distribution
    populations = [data['population'] for data in location_data.values() if data['population'] > 0]
    if populations:
        print(f"Population range: {min(populations):,.0f} to {max(populations):,.0f}")
        print(f"Average population: {sum(populations) / len(populations):,.0f}")


def debug_intents():
    """Debug the intents to understand classification"""

    print("\n=== DEBUGGING INTENTS ===")

    try:
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        print(f"✓ Loaded intents with {len(intents['intents'])} categories")
    except FileNotFoundError:
        print("❌ intents.json not found. Run prepare_census_data.py first.")
        return

    # Show intent categories
    for intent in intents['intents']:
        tag = intent['tag']
        patterns = len(intent['patterns'])
        responses = len(intent['responses'])
        print(f"  {tag}: {patterns} patterns, {responses} responses")

        # Show sample patterns
        if patterns > 0:
            print(f"    Sample patterns: {intent['patterns'][:2]}")


def test_location_matching():
    """Test location matching with sample queries"""

    print("\n=== TESTING LOCATION MATCHING ===")

    # Load data
    try:
        with open('location_data.json', 'r', encoding='utf-8') as f:
            location_data = json.load(f)
    except FileNotFoundError:
        print("❌ location_data.json not found.")
        return

    test_queries = [
        "tell me about pulwama",
        "what is the male population in kangan",
        "male population in punch",
        "population of poonch",
        "demographics of srinagar"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Simple location extraction
        query_lower = query.lower()
        query_words = [word for word in query_lower.split()
                       if word not in ['what', 'is', 'the', 'in', 'of', 'about', 'tell', 'me', 'population', 'male']]

        print(f"  Extract words: {query_words}")

        # Try to find matches
        matches = []
        for location in location_data.keys():
            for word in query_words:
                if word in location.lower():
                    matches.append(location)
                    break

        if matches:
            print(f"  Found matches: {matches[:3]}")
            for match in matches[:1]:  # Show data for first match
                data = location_data[match]
                print(f"    {match}: Pop={data['population']:,.0f}, Males={data['males']:,.0f}")
        else:
            print(f"  No matches found")


def analyze_csv_structure():
    """Analyze the original CSV structure"""

    print("\n=== ANALYZING CSV STRUCTURE ===")

    try:
        df = pd.read_csv('feature_engineered.csv')
        print(f"✓ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Clean column names
        df.columns = df.columns.str.replace('\n', '').str.replace('\r', '').str.strip()

        print(f"\nColumns: {list(df.columns)}")

        # Show unique location types
        if 'Name' in df.columns:
            unique_names = df['Name'].dropna().unique()
            print(f"\nTotal unique locations: {len(unique_names)}")

            # Sample names
            print("Sample location names:")
            for name in unique_names[:10]:
                print(f"  - {name}")

        # Check for Kashmir region specifically
        if 'Name' in df.columns:
            kashmir_locations = df[df['Name'].str.contains('Kashmir|Pulwama|Kangan|Punch|Poonch', case=False, na=False)]
            print(f"\nKashmir region locations found: {len(kashmir_locations)}")
            for _, row in kashmir_locations.head().iterrows():
                print(f"  - {row['Name']}: Pop={row.get('Population (Persons)', 'N/A')}")

    except FileNotFoundError:
        print("❌ feature_engineered.csv not found")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")


if __name__ == "__main__":
    debug_location_data()
    debug_intents()
    test_location_matching()
    analyze_csv_structure()