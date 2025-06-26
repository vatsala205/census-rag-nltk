import pandas as pd
import json

# Load the census CSV file
df = pd.read_csv('feature_engineered.csv')  # Your actual CSV file

# Clean column names - handle line breaks and spaces
df.columns = df.columns.str.replace('\n', '').str.replace('\r', '').str.strip()
df.columns = df.columns.str.replace('Sub District Cod\ne', 'Sub District Code')

print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
print("Column names:", list(df.columns))

# Create intents for different types of demographic queries
intents = {"intents": []}

# Define intent categories and generate patterns/responses
intent_definitions = {
    "population_query": {
        "patterns": [
            "What is the population of {location}",
            "How many people live in {location}",
            "Population of {location}",
            "Tell me about population in {location}",
            "How populated is {location}",
            "Population count for {location}",
            "Demographics of {location}",
            "Total population in {location}"
        ],
        "base_response": "The population of {location} is {population:,} with {males:,} males and {females:,} females."
    },

    "area_query": {
        "patterns": [
            "What is the area of {location}",
            "How big is {location}",
            "Area of {location}",
            "Size of {location}",
            "Land area in {location}",
            "Square kilometers of {location}",
            "How much area does {location} cover"
        ],
        "base_response": "The area of {location} is {area} square kilometers with a population density of {density} people per sq km."
    },

    "household_query": {
        "patterns": [
            "How many households in {location}",
            "Number of households in {location}",
            "Household count for {location}",
            "Families in {location}",
            "Total households in {location}"
        ],
        "base_response": "There are {households:,} households in {location}."
    },

    "rural_urban_query": {
        "patterns": [
            "Rural urban ratio of {location}",
            "What is the rural urban distribution in {location}",
            "Rural vs urban population in {location}",
            "How much is rural and urban in {location}"
        ],
        "base_response": "The rural-urban ratio in {location} is {ratio}."
    },

    "villages_towns_query": {
        "patterns": [
            "How many villages in {location}",
            "Number of towns in {location}",
            "Villages and towns in {location}",
            "Tell me about villages in {location}",
            "How many inhabited villages in {location}"
        ],
        "base_response": "In {location}, there are {villages:,} inhabited villages, {uninhabited:,} uninhabited villages, and {towns:,} towns."
    },

    "comparison_query": {
        "patterns": [
            "Compare population of states",
            "Which state has highest population",
            "Most populated state in India",
            "Least populated state",
            "State wise population comparison",
            "Rank states by population"
        ],
        "base_response": "I can help you compare demographic data across different regions."
    },

    "general_info": {
        "patterns": [
            "What data do you have",
            "What can you tell me",
            "Help me with demographics",
            "What information is available",
            "Tell me about Indian demographics",
            "What kind of data can you provide"
        ],
        "base_response": "I have comprehensive Indian census data including population, area, households, rural-urban distribution, and village/town information for different states, districts, and sub-districts."
    }
}

# Generate patterns for each location in the dataset
locations = []
location_data = {}

# Process the data to extract unique locations and their data
for _, row in df.iterrows():
    location_name = row['Name'].strip()
    if location_name and location_name not in ['INDIA @&', 'INDIA $']:
        locations.append(location_name)
        location_data[location_name] = {
            'population': row.get('Population (Persons)', 0),
            'males': row.get('Population (Males)', 0),
            'females': row.get('Population (Females)', 0),
            'area': row.get('Area (sq km)', 0),
            'density': row.get('Population Density', 0),
            'households': row.get('Number of Households', 0),
            'villages': row.get('Number of Villages (Inhabited)', 0),
            'uninhabited': row.get('Number of Villages (Uninhabited)', 0),
            'towns': row.get('Number of Towns', 0),
            'ratio': row.get('Rural-Urban Ratio', 0)
        }

# Create intents with actual location names
for intent_tag, intent_data in intent_definitions.items():
    patterns = []
    responses = []

    if intent_tag in ['population_query', 'area_query', 'household_query', 'rural_urban_query', 'villages_towns_query']:
        # Generate patterns for each location
        for location in locations[:100]:  # Limit to first 100 locations to keep manageable
            for pattern_template in intent_data['patterns']:
                pattern = pattern_template.replace('{location}', location)
                patterns.append(pattern)

        # Generate generic patterns without specific locations
        generic_patterns = [p.replace(' {location}', '').replace('{location} ', '').replace('{location}', 'a place')
                            for p in intent_data['patterns']]
        patterns.extend(generic_patterns)

        # Add some sample responses
        responses = [
            intent_data['base_response'].replace('{location}', 'the requested location')
                .replace('{population:,}', 'X')
                .replace('{males:,}', 'Y')
                .replace('{females:,}', 'Z')
                .replace('{area}', 'A')
                .replace('{density}', 'B')
                .replace('{households:,}', 'C')
                .replace('{villages:,}', 'D')
                .replace('{uninhabited:,}', 'E')
                .replace('{towns:,}', 'F')
                .replace('{ratio}', 'G'),
            "I can provide demographic information for that location.",
            "Let me help you with the demographic data for that area.",
            "I have census information available for that region."
        ]
    else:
        # For general queries, use patterns as-is
        patterns = intent_data['patterns']
        responses = [intent_data['base_response']]

    intent = {
        "tag": intent_tag,
        "patterns": patterns[:200],  # Limit patterns to prevent oversized file
        "responses": responses
    }
    intents["intents"].append(intent)

# Add a fallback intent
fallback_intent = {
    "tag": "fallback",
    "patterns": [
        "I don't understand",
        "What?",
        "Can you help me",
        "I need help",
        "Hello",
        "Hi"
    ],
    "responses": [
        "I'm here to help with Indian demographic data. You can ask about population, area, households, or rural-urban distribution for different states and districts.",
        "I can provide census information for Indian states, districts, and sub-districts. What would you like to know?",
        "Ask me about population, area, households, villages, or towns in different parts of India.",
        "I have demographic data for India. Try asking about specific locations or comparisons."
    ]
}
intents["intents"].append(fallback_intent)

# Save intents to JSON file
with open('intents.json', 'w', encoding='utf-8') as f:
    json.dump(intents, f, indent=4, ensure_ascii=False)

# Save location data for use in the chatbot
with open('location_data.json', 'w', encoding='utf-8') as f:
    json.dump(location_data, f, indent=4, ensure_ascii=False)

print(f"Created intents.json with {len(intents['intents'])} intents.")
print(f"Processed {len(locations)} unique locations.")
print("Location data saved to location_data.json")