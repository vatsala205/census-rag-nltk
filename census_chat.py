import json
import random
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Download nltk data if not already done
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

lemmatizer = WordNetLemmatizer()

# Global variables for model and data
intents = None
location_data = None
words = None
classes = None
model = None


def load_chatbot_data():
    """Load all required data and model"""
    global intents, location_data, words, classes, model

    try:
        # Load intents
        with open('intents.json', 'r', encoding='utf-8') as file:
            intents = json.load(file)

        # Load location data
        with open('location_data.json', 'r', encoding='utf-8') as file:
            location_data = json.load(file)

        # Load training data
        with open('training_data.pkl', 'rb') as f:
            words, classes, X_train, y_train = pickle.load(f)

        # Load model
        model = load_model('chatbot_model.h5')

        return True
    except Exception as e:
        print(f"Error loading chatbot data: {e}")
        return False


# Load data when module is imported
load_chatbot_data()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    if model is None or words is None or classes is None:
        return []

    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def extract_demographic_type(text):
    """Extract what type of demographic information is being asked"""
    text_lower = text.lower()

    # Define patterns for different types of queries
    demographic_patterns = {
        'female_population': ['female population', 'women population', 'female count', 'number of females',
                              'how many females', 'females in'],
        'male_population': ['male population', 'men population', 'male count', 'number of males', 'how many males',
                            'males in'],
        'total_population': ['total population', 'population', 'people', 'inhabitants', 'residents'],
        'area': ['area', 'size', 'how big', 'square kilometers', 'sq km'],
        'households': ['households', 'families', 'homes', 'houses'],
        'density': ['density', 'population density', 'people per sq km'],
        'villages': ['villages', 'inhabited villages', 'number of villages'],
        'towns': ['towns', 'number of towns', 'urban centers'],
        'ratio': ['ratio', 'rural urban ratio', 'rural-urban ratio']
    }

    for demo_type, patterns in demographic_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                return demo_type

    return 'general'


def extract_location_from_query(text):
    """
    Enhanced location extraction that handles specific demographic queries
    """
    if not location_data:
        return None, None

    # Convert to lowercase for matching
    text_lower = text.lower().strip()

    # Remove common query words but keep location indicators
    query_words_to_remove = ['what', 'is', 'the', 'about', 'tell', 'me', 'how', 'many', 'much', 'show', 'give', 'in',
                             'of', 'for']
    demographic_words = ['population', 'area', 'households', 'male', 'female', 'people', 'villages', 'towns', 'density',
                         'ratio', 'women', 'men', 'families']

    # Split text into words
    text_words = text_lower.split()

    # Extract potential location words (not demographic or query words)
    location_words = []
    for word in text_words:
        if word not in query_words_to_remove and word not in demographic_words and len(word) > 2:
            location_words.append(word)

    # Method 1: Direct exact match with Total suffix priority
    for location_key in location_data.keys():
        # Check for exact match with Total suffix
        if location_key.lower().endswith(' total'):
            base_location = location_key[:-6].lower()  # Remove ' Total'
            # Check if base location matches any combination of location words
            if base_location in text_lower or any(word in base_location for word in location_words):
                return location_key, location_data[location_key]

    # Method 2: Check individual words against location names
    potential_matches = []

    for word in location_words:
        if len(word) < 3:  # Skip very short words
            continue

        for location_key in location_data.keys():
            location_lower = location_key.lower()

            # Extract base location name (without Total/Rural/Urban)
            base_location = location_lower
            if location_lower.endswith(' total'):
                base_location = location_lower[:-6]
            elif location_lower.endswith(' rural'):
                base_location = location_lower[:-6]
            elif location_lower.endswith(' urban'):
                base_location = location_lower[:-6]

            # Check exact word match or if word is part of location name
            if (word == base_location or
                    word in base_location.split() or
                    base_location.startswith(word) or
                    any(loc_word.startswith(word) for loc_word in base_location.split())):

                # Prioritize Total data
                if location_key.endswith(' Total'):
                    return location_key, location_data[location_key]
                else:
                    potential_matches.append((location_key, location_data[location_key]))

    # Method 3: Fuzzy matching for common abbreviations and variations
    location_variations = {
        'delhi': ['delhi', 'new delhi'],
        'mumbai': ['mumbai', 'bombay'],
        'kolkata': ['kolkata', 'calcutta'],
        'chennai': ['chennai', 'madras'],
        'bangalore': ['bangalore', 'bengaluru'],
        'hyderabad': ['hyderabad', 'hyd'],
        'pune': ['pune', 'poona'],
        'goa': ['goa'],
        'kerala': ['kerala'],
        'gujarat': ['gujarat'],
        'rajasthan': ['rajasthan'],
        'punjab': ['punjab'],
        'haryana': ['haryana'],
        'up': ['uttar pradesh', 'u.p.', 'up'],
        'mp': ['madhya pradesh', 'm.p.', 'mp']
    }

    for word in location_words:
        for standard_name, variations in location_variations.items():
            if word in variations:
                # Look for the standard name in location data
                for location_key in location_data.keys():
                    if location_key.lower().startswith(standard_name) and location_key.endswith(' Total'):
                        return location_key, location_data[location_key]

    # Method 4: If we have potential matches but no Total, look for Total version
    if potential_matches:
        # Extract base names from potential matches and look for Total version
        for match_key, match_data in potential_matches:
            base_name = match_key
            if match_key.endswith(' Rural'):
                base_name = match_key[:-6]
            elif match_key.endswith(' Urban'):
                base_name = match_key[:-6]

            # Look for Total version
            total_key = f"{base_name} Total"
            if total_key in location_data:
                return total_key, location_data[total_key]

        # If no Total found, return first match
        return potential_matches[0]

    return None, None


def format_specific_demographic_response(demo_type, location_key, data):
    """Format response for specific demographic queries"""
    if not data:
        return f"Sorry, I don't have demographic data for the requested location."

    # Extract clean location name
    location_name = location_key
    if location_key.endswith(' Total'):
        location_name = location_key[:-6]
    elif location_key.endswith(' Rural'):
        location_name = location_key[:-6]
    elif location_key.endswith(' Urban'):
        location_name = location_key[:-6]

    if demo_type == 'female_population':
        females = data.get('females', 0)
        total_pop = data.get('population', 0)
        if females > 0:
            percentage = (females / total_pop * 100) if total_pop > 0 else 0
            return f"The female population in {location_name} is {females:,} ({percentage:.1f}% of total population)."
        else:
            return f"No female population data available for {location_name}."

    elif demo_type == 'male_population':
        males = data.get('males', 0)
        total_pop = data.get('population', 0)
        if males > 0:
            percentage = (males / total_pop * 100) if total_pop > 0 else 0
            return f"The male population in {location_name} is {males:,} ({percentage:.1f}% of total population)."
        else:
            return f"No male population data available for {location_name}."

    elif demo_type == 'total_population':
        total_pop = data.get('population', 0)
        if total_pop > 0:
            return f"The total population of {location_name} is {total_pop:,}."
        else:
            return f"No population data available for {location_name}."

    elif demo_type == 'area':
        area = data.get('area', 0)
        if area > 0:
            return f"The area of {location_name} is {area:,} square kilometers."
        else:
            return f"No area data available for {location_name}."

    elif demo_type == 'households':
        households = data.get('households', 0)
        if households > 0:
            return f"There are {households:,} households in {location_name}."
        else:
            return f"No household data available for {location_name}."

    elif demo_type == 'density':
        density = data.get('density', 0)
        if density > 0:
            return f"The population density of {location_name} is {density:,} people per square kilometer."
        else:
            return f"No population density data available for {location_name}."

    elif demo_type == 'villages':
        villages = data.get('villages', 0)
        if villages > 0:
            return f"There are {villages:,} inhabited villages in {location_name}."
        else:
            return f"No village data available for {location_name}."

    elif demo_type == 'towns':
        towns = data.get('towns', 0)
        if towns > 0:
            return f"There are {towns:,} towns in {location_name}."
        else:
            return f"No town data available for {location_name}."

    elif demo_type == 'ratio':
        ratio = data.get('ratio', 0)
        if ratio > 0:
            return f"The rural-urban ratio in {location_name} is {ratio:.2f}."
        else:
            return f"No rural-urban ratio data available for {location_name}."

    else:
        # General response
        return format_demographic_response('general', location_key, data)


def format_demographic_response(intent_tag, location_key, data):
    """Format demographic data into readable response - original function"""
    if not data:
        return f"Sorry, I don't have demographic data for the requested location."

    # Extract clean location name
    location_name = location_key
    if location_key.endswith(' Total'):
        location_name = location_key[:-6]
    elif location_key.endswith(' Rural'):
        location_name = location_key[:-6]
    elif location_key.endswith(' Urban'):
        location_name = location_key[:-6]

    if intent_tag == "population_query":
        total_pop = data.get('population', 0)
        males = data.get('males', 0)
        females = data.get('females', 0)

        if total_pop > 0:
            male_pct = (males / total_pop * 100) if total_pop > 0 else 0
            female_pct = (females / total_pop * 100) if total_pop > 0 else 0
            return f"The population of {location_name} is {total_pop:,} with {males:,} males ({male_pct:.1f}%) and {females:,} females ({female_pct:.1f}%)."
        else:
            return f"{location_name} appears to have no recorded population data."

    elif intent_tag == "area_query":
        area = data.get('area', 0)
        density = data.get('density', 0)
        if area > 0:
            return f"The area of {location_name} is {area:,} square kilometers with a population density of {density:,} people per sq km."
        else:
            return f"No area data available for {location_name}."

    elif intent_tag == "household_query":
        households = data.get('households', 0)
        if households > 0:
            return f"There are {households:,} households in {location_name}."
        else:
            return f"No household data available for {location_name}."

    elif intent_tag == "rural_urban_query":
        ratio = data.get('ratio', 0)
        if ratio > 0:
            return f"The rural-urban ratio in {location_name} is {ratio:.2f} (meaning {ratio:.2f} rural people for every 1 urban person)."
        else:
            return f"No rural-urban ratio data available for {location_name}."

    elif intent_tag == "villages_towns_query":
        villages = data.get('villages', 0)
        uninhabited = data.get('uninhabited', 0)
        towns = data.get('towns', 0)
        return f"In {location_name}, there are {villages:,} inhabited villages, {uninhabited:,} uninhabited villages, and {towns:,} towns."

    else:
        # General overview
        pop = data.get('population', 0)
        area = data.get('area', 0)
        households = data.get('households', 0)
        if pop > 0:
            return f"Here's the demographic overview for {location_name}: Population: {pop:,}, Area: {area:,} sq km, Households: {households:,}"
        else:
            return f"Limited demographic data available for {location_name}."


def suggest_similar_locations(user_input, max_suggestions=3):
    """Suggest similar location names based on user input - prioritize Total entries"""
    if not location_data:
        return []

    suggestions = []
    query_words = user_input.lower().split()

    # First, look for Total entries
    for location in location_data.keys():
        if location.endswith(' Total'):
            base_location = location[:-6].lower()
            for word in query_words:
                if len(word) > 2 and (word in base_location or base_location.startswith(word)):
                    clean_name = location[:-6]  # Remove ' Total'
                    if clean_name not in suggestions:
                        suggestions.append(clean_name)
                        break

    # If not enough suggestions, look at all entries
    if len(suggestions) < max_suggestions:
        for location in location_data.keys():
            location_lower = location.lower()
            # Extract base name
            base_name = location
            if location.endswith(' Total') or location.endswith(' Rural') or location.endswith(' Urban'):
                base_name = location[:-6]

            for word in query_words:
                if len(word) > 2 and (word in location_lower or location_lower.startswith(word)):
                    if base_name not in suggestions:
                        suggestions.append(base_name)
                        break

    return suggestions[:max_suggestions]


def get_response(intents_list, intents_json, user_input):
    """Enhanced response generation that handles specific demographic queries"""

    # First, check if this is a specific demographic query
    demo_type = extract_demographic_type(user_input)
    location_key, data = extract_location_from_query(user_input)

    # If we found a location and it's asking for specific demographic info
    if location_key and data and demo_type != 'general':
        if data.get('population', 0) == 0 and data.get('area', 0) == 0:
            return f"I found the location in the database, but it appears to have no recorded data."
        return format_specific_demographic_response(demo_type, location_key, data)

    # Original intent-based processing
    if not intents_list:
        # If no intent but we have location info, try to provide general info
        if location_key and data:
            return format_specific_demographic_response('general', location_key, data)
        return "Sorry, I didn't understand that. You can ask me about population, area, households, or villages/towns for different places in India."

    intent_tag = intents_list[0]['intent']
    confidence = float(intents_list[0]['probability'])

    # Check if this is a location-specific query
    location_specific_intents = ['population_query', 'area_query', 'household_query', 'rural_urban_query',
                                 'villages_towns_query']

    if intent_tag in location_specific_intents:
        if location_key and data:
            # Check if we have valid data
            if data.get('population', 0) == 0 and data.get('area', 0) == 0:
                return f"I found the location in the database, but it appears to have no recorded data."

            return format_demographic_response(intent_tag, location_key, data)
        else:
            # Try to suggest similar locations
            similar_locations = suggest_similar_locations(user_input)

            if similar_locations:
                return f"I couldn't find that exact location. Did you mean one of these? {', '.join(similar_locations)}. Please try again with the exact spelling."
            else:
                return f"I couldn't identify the specific location in your query. Could you please check the spelling or try a different location name? You can ask about states, districts, or cities in India."

    elif intent_tag == "comparison_query":
        if not location_data:
            return "Sorry, location data is not available right now."

        # For comparison queries, provide top populated regions (Total data only)
        total_regions = {k: v for k, v in location_data.items() if k.endswith(' Total')}
        top_regions = sorted(total_regions.items(), key=lambda x: x[1].get('population', 0), reverse=True)[:5]
        response = "Here are the top 5 most populated regions:\n"
        for i, (region_key, data) in enumerate(top_regions, 1):
            pop = data.get('population', 0)
            region_name = region_key[:-6]  # Remove ' Total'
            if pop > 0:
                response += f"{i}. {region_name}: {pop:,} people\n"
        return response

    else:
        # For general queries, use standard responses
        if intents_json and 'intents' in intents_json:
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == intent_tag:
                    return random.choice(i['responses'])

    return "I'm here to help with Indian demographic data. Ask me about population, area, households, or other statistics for different locations in India."


def chatbot_response(text):
    """Main chatbot response function"""
    if not all([intents, location_data, words, classes, model]):
        return "Sorry, the chatbot is not properly initialized. Please try again later."

    # Check for exit commands
    exit_words = ['quit', 'exit', 'bye', 'goodbye', 'stop']
    if any(word in text.lower() for word in exit_words):
        return "Thank you for using the Indian Demographics Chatbot! Goodbye! 👋"

    try:
        # Process the query first
        processed_text = process_query(text)

        # Get intent predictions
        ints = predict_class(processed_text)

        # Generate response (now handles specific demographic queries)
        res = get_response(ints, intents, processed_text)
        return res
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return "Sorry, I encountered an error processing your request. Please try again."


def process_query(text):
    """Enhanced query processing for better understanding"""
    if not text:
        return ""

    text = text.lower().strip()

    # Replace common synonyms and variations
    replacements = {
        'how many people': 'population',
        'how big': 'area',
        'size of': 'area of',
        'families': 'households',
        'rural urban': 'rural-urban ratio',
        'number of people': 'population',
        'how much area': 'area',
        'women': 'female',
        'men': 'male',
        'ladies': 'female',
        'gentlemen': 'male'
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# Test function for debugging
def test_chatbot():
    """Test function to check if chatbot is working with specific queries"""
    test_queries = [
        "female population in delhi",
        "male population in goa",
        "what is the population of goa",
        "population of delhi",
        "area of mumbai",
        "households in kerala",
        "how many women in punjab",
        "men in haryana",
        "total people in rajasthan"
    ]

    print("Testing enhanced chatbot responses:")
    for query in test_queries:
        processed = process_query(query)
        response = chatbot_response(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
        print("-" * 50)


# For command line testing
if __name__ == "__main__":
    if not all([intents, location_data, words, classes, model]):
        print("Error: Could not load required files. Make sure these files exist:")
        print("- intents.json")
        print("- location_data.json")
        print("- training_data.pkl")
        print("- chatbot_model.h5")
        exit(1)

    print("Enhanced Indian Demographics Chatbot")
    print("Ask me about population, area, households, villages, or towns in different parts of India!")
    print("Examples:")
    print("- 'female population in Delhi'")
    print("- 'male population in Mumbai'")
    print("- 'how many women in Punjab'")
    print("- 'area of Kerala'")
    print("(Type 'quit' to stop)\n")

    while True:
        message = input("You: ")
        if message.lower() in ['quit', 'exit', 'bye', 'goodbye', 'stop']:
            print("Bot: Thank you for using the Indian Demographics Chatbot! Goodbye! 👋")
            break

        response = chatbot_response(message)
        print("Bot:", response)
        print()

# MULTIPLE OUTPUTS
# import json
# import random
# import numpy as np
# import nltk
# import pickle
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # Download nltk data if not already done
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('wordnet', quiet=True)
# except:
#     pass
#
# lemmatizer = WordNetLemmatizer()
#
# # Global variables for model and data
# intents = None
# location_data = None
# words = None
# classes = None
# model = None
#
#
# def load_chatbot_data():
#     """Load all required data and model"""
#     global intents, location_data, words, classes, model
#
#     try:
#         # Load intents
#         with open('intents.json', 'r', encoding='utf-8') as file:
#             intents = json.load(file)
#
#         # Load location data
#         with open('location_data.json', 'r', encoding='utf-8') as file:
#             location_data = json.load(file)
#
#         # Load training data
#         with open('training_data.pkl', 'rb') as f:
#             words, classes, X_train, y_train = pickle.load(f)
#
#         # Load model
#         model = load_model('chatbot_model.h5')
#
#         return True
#     except Exception as e:
#         print(f"Error loading chatbot data: {e}")
#         return False
#
#
# # Load data when module is imported
# load_chatbot_data()
#
#
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words
#
#
# def bag_of_words(sentence, words):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#     return np.array(bag)
#
#
# def predict_class(sentence):
#     if model is None or words is None or classes is None:
#         return []
#
#     bow = bag_of_words(sentence, words)
#     res = model.predict(np.array([bow]), verbose=0)[0]
#     ERROR_THRESHOLD = 0.25
#     results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list
#
#
# def extract_location_from_query(text):
#     """
#     Extract location name from user query - PRIORITIZES TOTAL DATA
#     """
#     if not location_data:
#         return None, None
#
#     # Convert to lowercase for matching
#     text_lower = text.lower().strip()
#
#     # Remove common query words
#     query_words_to_remove = ['what', 'is', 'the', 'about', 'tell', 'me', 'how', 'many', 'much', 'show', 'give']
#     location_indicators = ['population', 'area', 'households', 'male', 'female', 'people', 'in', 'of', 'for',
#                            'villages', 'towns']
#
#     # Split text into words
#     text_words = text_lower.split()
#     filtered_words = [word for word in text_words if
#                       word not in query_words_to_remove and word not in location_indicators]
#
#     # Method 1: Direct match - prioritize Total
#     for location_key in location_data.keys():
#         # Check for exact match with Total suffix
#         if location_key.lower().endswith(' total'):
#             base_location = location_key[:-6].lower()  # Remove ' Total'
#             if base_location == text_lower or base_location in text_lower:
#                 return location_key, location_data[location_key]
#
#     # Method 2: Check if any word in query matches location name (prioritize Total)
#     potential_matches = []
#
#     for word in filtered_words:
#         if len(word) < 3:  # Skip very short words
#             continue
#
#         for location_key in location_data.keys():
#             location_lower = location_key.lower()
#
#             # Extract base location name (without Total/Rural/Urban)
#             base_location = location_lower
#             if location_lower.endswith(' total'):
#                 base_location = location_lower[:-6]
#             elif location_lower.endswith(' rural'):
#                 base_location = location_lower[:-6]
#             elif location_lower.endswith(' urban'):
#                 base_location = location_lower[:-6]
#
#             # Check if word matches the base location
#             if word == base_location or word in base_location.split():
#                 # Prioritize Total data
#                 if location_key.endswith(' Total'):
#                     return location_key, location_data[location_key]
#                 else:
#                     potential_matches.append((location_key, location_data[location_key]))
#
#     # Method 3: Partial matching with Total priority
#     for word in filtered_words:
#         if len(word) < 3:
#             continue
#
#         for location_key in location_data.keys():
#             location_lower = location_key.lower()
#
#             # Extract base location name
#             base_location = location_lower
#             if location_lower.endswith(' total'):
#                 base_location = location_lower[:-6]
#             elif location_lower.endswith(' rural'):
#                 base_location = location_lower[:-6]
#             elif location_lower.endswith(' urban'):
#                 base_location = location_lower[:-6]
#
#             # Check for partial matches
#             if (word in base_location or base_location.startswith(word) or
#                     any(loc_word.startswith(word) for loc_word in base_location.split())):
#
#                 # Prioritize Total data
#                 if location_key.endswith(' Total'):
#                     return location_key, location_data[location_key]
#                 else:
#                     potential_matches.append((location_key, location_data[location_key]))
#
#     # If we have potential matches but no Total, look for Total version
#     if potential_matches:
#         # Extract base names from potential matches
#         for match_key, match_data in potential_matches:
#             base_name = match_key
#             if match_key.endswith(' Rural'):
#                 base_name = match_key[:-6]
#             elif match_key.endswith(' Urban'):
#                 base_name = match_key[:-6]
#
#             # Look for Total version
#             total_key = f"{base_name} Total"
#             if total_key in location_data:
#                 return total_key, location_data[total_key]
#
#         # If no Total found, return first match
#         return potential_matches[0]
#
#     return None, None
#
#
# def format_demographic_response(intent_tag, location_key, data):
#     """Format demographic data into readable response"""
#     if not data:
#         return f"Sorry, I don't have demographic data for the requested location."
#
#     # Extract clean location name (remove Total/Rural/Urban suffix)
#     location_name = location_key
#     if location_key.endswith(' Total'):
#         location_name = location_key[:-6]
#     elif location_key.endswith(' Rural'):
#         location_name = location_key[:-6]
#     elif location_key.endswith(' Urban'):
#         location_name = location_key[:-6]
#
#     if intent_tag == "population_query":
#         total_pop = data.get('population', 0)
#         males = data.get('males', 0)
#         females = data.get('females', 0)
#
#         if total_pop > 0:
#             male_pct = (males / total_pop * 100) if total_pop > 0 else 0
#             female_pct = (females / total_pop * 100) if total_pop > 0 else 0
#             return f"The population of {location_name} is {total_pop:,} with {males:,} males ({male_pct:.1f}%) and {females:,} females ({female_pct:.1f}%)."
#         else:
#             return f"{location_name} appears to have no recorded population data."
#
#     elif intent_tag == "area_query":
#         area = data.get('area', 0)
#         density = data.get('density', 0)
#         if area > 0:
#             return f"The area of {location_name} is {area:,} square kilometers with a population density of {density:,} people per sq km."
#         else:
#             return f"No area data available for {location_name}."
#
#     elif intent_tag == "household_query":
#         households = data.get('households', 0)
#         if households > 0:
#             return f"There are {households:,} households in {location_name}."
#         else:
#             return f"No household data available for {location_name}."
#
#     elif intent_tag == "rural_urban_query":
#         ratio = data.get('ratio', 0)
#         if ratio > 0:
#             return f"The rural-urban ratio in {location_name} is {ratio:.2f} (meaning {ratio:.2f} rural people for every 1 urban person)."
#         else:
#             return f"No rural-urban ratio data available for {location_name}."
#
#     elif intent_tag == "villages_towns_query":
#         villages = data.get('villages', 0)
#         uninhabited = data.get('uninhabited', 0)
#         towns = data.get('towns', 0)
#         return f"In {location_name}, there are {villages:,} inhabited villages, {uninhabited:,} uninhabited villages, and {towns:,} towns."
#
#     else:
#         # General overview - prioritize meaningful data
#         pop = data.get('population', 0)
#         area = data.get('area', 0)
#         households = data.get('households', 0)
#         if pop > 0:
#             return f"Here's the demographic overview for {location_name}: Population: {pop:,}, Area: {area:,} sq km, Households: {households:,}"
#         else:
#             return f"Limited demographic data available for {location_name}."
#
#
# def suggest_similar_locations(user_input, max_suggestions=3):
#     """Suggest similar location names based on user input - prioritize Total entries"""
#     if not location_data:
#         return []
#
#     suggestions = []
#     query_words = user_input.lower().split()
#
#     # First, look for Total entries
#     for location in location_data.keys():
#         if location.endswith(' Total'):
#             base_location = location[:-6].lower()
#             for word in query_words:
#                 if len(word) > 2 and (word in base_location or base_location.startswith(word)):
#                     clean_name = location[:-6]  # Remove ' Total'
#                     if clean_name not in suggestions:
#                         suggestions.append(clean_name)
#                         break
#
#     # If not enough suggestions, look at all entries
#     if len(suggestions) < max_suggestions:
#         for location in location_data.keys():
#             location_lower = location.lower()
#             # Extract base name
#             base_name = location
#             if location.endswith(' Total') or location.endswith(' Rural') or location.endswith(' Urban'):
#                 base_name = location[:-6]
#
#             for word in query_words:
#                 if len(word) > 2 and (word in location_lower or location_lower.startswith(word)):
#                     if base_name not in suggestions:
#                         suggestions.append(base_name)
#                         break
#
#     return suggestions[:max_suggestions]
#
#
# def get_response(intents_list, intents_json, user_input):
#     """Generate response based on intent and user input"""
#     if not intents_list:
#         return "Sorry, I didn't understand that. You can ask me about population, area, households, or villages/towns for different places in India."
#
#     intent_tag = intents_list[0]['intent']
#     confidence = float(intents_list[0]['probability'])
#
#     # Check if this is a location-specific query
#     location_specific_intents = ['population_query', 'area_query', 'household_query', 'rural_urban_query',
#                                  'villages_towns_query']
#
#     if intent_tag in location_specific_intents:
#         location_key, data = extract_location_from_query(user_input)
#
#         if location_key and data:
#             # Check if we have valid data
#             if data.get('population', 0) == 0 and data.get('area', 0) == 0:
#                 return f"I found the location in the database, but it appears to have no recorded data."
#
#             return format_demographic_response(intent_tag, location_key, data)
#         else:
#             # Try to suggest similar locations
#             similar_locations = suggest_similar_locations(user_input)
#
#             if similar_locations:
#                 return f"I couldn't find that exact location. Did you mean one of these? {', '.join(similar_locations)}. Please try again with the exact spelling."
#             else:
#                 return f"I couldn't identify the specific location in your query. Could you please check the spelling or try a different location name? You can ask about states, districts, or cities in India."
#
#     elif intent_tag == "comparison_query":
#         if not location_data:
#             return "Sorry, location data is not available right now."
#
#         # For comparison queries, provide top populated regions (Total data only)
#         total_regions = {k: v for k, v in location_data.items() if k.endswith(' Total')}
#         top_regions = sorted(total_regions.items(), key=lambda x: x[1].get('population', 0), reverse=True)[:5]
#         response = "Here are the top 5 most populated regions:\n"
#         for i, (region_key, data) in enumerate(top_regions, 1):
#             pop = data.get('population', 0)
#             region_name = region_key[:-6]  # Remove ' Total'
#             if pop > 0:
#                 response += f"{i}. {region_name}: {pop:,} people\n"
#         return response
#
#     else:
#         # For general queries, use standard responses
#         if intents_json and 'intents' in intents_json:
#             list_of_intents = intents_json['intents']
#             for i in list_of_intents:
#                 if i['tag'] == intent_tag:
#                     return random.choice(i['responses'])
#
#     return "I'm here to help with Indian demographic data. Ask me about population, area, households, or other statistics for different locations in India."
#
#
# def chatbot_response(text):
#     """Main chatbot response function"""
#     if not all([intents, location_data, words, classes, model]):
#         return "Sorry, the chatbot is not properly initialized. Please try again later."
#
#     # Check for exit commands
#     exit_words = ['quit', 'exit', 'bye', 'goodbye', 'stop']
#     if any(word in text.lower() for word in exit_words):
#         return "Thank you for using the Indian Demographics Chatbot! Goodbye! 👋"
#
#     try:
#         ints = predict_class(text)
#         res = get_response(ints, intents, text)
#         return res
#     except Exception as e:
#         return "Sorry, I encountered an error processing your request. Please try again."
#
#
# def process_query(text):
#     """Process and enhance user queries for better understanding"""
#     if not text:
#         return ""
#
#     text = text.lower().strip()
#
#     # Replace common synonyms
#     replacements = {
#         'how many people': 'population',
#         'how big': 'area',
#         'size of': 'area of',
#         'families': 'households',
#         'rural urban': 'rural-urban ratio',
#         'number of people': 'population',
#         'how much area': 'area'
#     }
#
#     for old, new in replacements.items():
#         text = text.replace(old, new)
#
#     return text
#
#
# # Test function for debugging
# def test_chatbot():
#     """Test function to check if chatbot is working"""
#     test_queries = [
#         "what is the population of goa",
#         "population of delhi",
#         "area of mumbai",
#         "households in kerala"
#     ]
#
#     print("Testing chatbot responses:")
#     for query in test_queries:
#         processed = process_query(query)
#         response = chatbot_response(processed)
#         print(f"Query: {query}")
#         print(f"Response: {response}")
#         print("-" * 50)
#
#
# # For command line testing
# if __name__ == "__main__":
#     if not all([intents, location_data, words, classes, model]):
#         print("Error: Could not load required files. Make sure these files exist:")
#         print("- intents.json")
#         print("- location_data.json")
#         print("- training_data.pkl")
#         print("- chatbot_model.h5")
#         exit(1)
#
#     print("Indian Demographics Chatbot")
#     print("Ask me about population, area, households, villages, or towns in different parts of India!")
#     print("(Type 'quit' to stop)\n")
#
#     while True:
#         message = input("You: ")
#         if message.lower() in ['quit', 'exit', 'bye', 'goodbye', 'stop']:
#             print("Bot: Thank you for using the Indian Demographics Chatbot! Goodbye! 👋")
#             break
#
#         processed_message = process_query(message)
#         response = chatbot_response(processed_message)
#         print("Bot:", response)
#         prin