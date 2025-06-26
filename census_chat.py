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


def extract_location_from_query(text):
    """Extract location name from user query - Web compatible version"""
    if not location_data:
        return None

    # Convert to lowercase for matching
    text_lower = text.lower().strip()

    # Remove common query words
    query_words_to_remove = ['what', 'is', 'the', 'about', 'tell', 'me', 'how', 'many', 'much', 'show', 'give']
    location_indicators = ['population', 'area', 'households', 'male', 'female', 'people', 'in', 'of', 'for']

    # Split text into words
    text_words = text_lower.split()
    filtered_words = [word for word in text_words if word not in query_words_to_remove]

    # Method 1: Direct exact match (case-insensitive)
    for location in location_data.keys():
        if location.lower() == text_lower:
            return location

    # Method 2: Check if location name appears in the text
    for location in location_data.keys():
        location_lower = location.lower()
        if location_lower in text_lower:
            return location

    # Method 3: Check individual words against location names
    for word in filtered_words:
        if word in location_indicators:
            continue

        for location in location_data.keys():
            location_lower = location.lower()
            # Check if the word matches the location exactly
            if word == location_lower:
                return location

            # Check if the word is part of a multi-word location name
            location_words = location_lower.split()
            if word in location_words:
                return location

    # Method 4: Fuzzy matching for partial names
    for word in filtered_words:
        if len(word) < 3 or word in location_indicators:
            continue

        for location in location_data.keys():
            location_lower = location.lower()
            location_words = location_lower.split()

            for loc_word in location_words:
                if len(loc_word) >= 3:
                    # Check if word starts with the same letters
                    if word.startswith(loc_word[:3]) or loc_word.startswith(word[:3]):
                        return location

                    # Check for substring matches
                    if word in loc_word or loc_word in word:
                        return location

    return None


def format_demographic_response(intent_tag, location, data):
    """Format demographic data into readable response"""
    if not data:
        return f"Sorry, I don't have demographic data for {location}."

    if intent_tag == "population_query":
        total_pop = data.get('population', 0)
        males = data.get('males', 0)
        females = data.get('females', 0)

        if total_pop > 0:
            return f"The population of {location} is {total_pop:,.0f} with {males:,.0f} males ({males / total_pop * 100:.1f}%) and {females:,.0f} females ({females / total_pop * 100:.1f}%)."
        else:
            return f"{location} appears to have no recorded population data."

    elif intent_tag == "area_query":
        area = data.get('area', 0)
        density = data.get('density', 0)
        if area > 0:
            return f"The area of {location} is {area:,.0f} square kilometers with a population density of {density:,.0f} people per sq km."
        else:
            return f"No area data available for {location}."

    elif intent_tag == "household_query":
        households = data.get('households', 0)
        if households > 0:
            return f"There are {households:,.0f} households in {location}."
        else:
            return f"No household data available for {location}."

    elif intent_tag == "rural_urban_query":
        ratio = data.get('ratio', 0)
        if ratio > 0:
            return f"The rural-urban ratio in {location} is {ratio:.2f} (meaning {ratio:.2f} rural people for every 1 urban person)."
        else:
            return f"No rural-urban ratio data available for {location}."

    elif intent_tag == "villages_towns_query":
        villages = data.get('villages', 0)
        uninhabited = data.get('uninhabited', 0)
        towns = data.get('towns', 0)
        return f"In {location}, there are {villages:,.0f} inhabited villages, {uninhabited:,.0f} uninhabited villages, and {towns:,.0f} towns."

    else:
        # General overview
        pop = data.get('population', 0)
        area = data.get('area', 0)
        households = data.get('households', 0)
        if pop > 0:
            return f"Here's the demographic overview for {location}: Population: {pop:,.0f}, Area: {area:,.0f} sq km, Households: {households:,.0f}"
        else:
            return f"Limited demographic data available for {location}."


def suggest_similar_locations(user_input, max_suggestions=3):
    """Suggest similar location names based on user input"""
    if not location_data:
        return []

    suggestions = []
    query_words = user_input.lower().split()

    for location in location_data.keys():
        location_lower = location.lower()
        for word in query_words:
            if len(word) > 2 and (word in location_lower or location_lower.startswith(word)):
                if location not in suggestions:
                    suggestions.append(location)
                    break

    return suggestions[:max_suggestions]


def get_response(intents_list, intents_json, user_input):
    """Generate response based on intent and user input"""
    if not intents_list:
        return "Sorry, I didn't understand that. You can ask me about population, area, households, or villages/towns for different places in India."

    intent_tag = intents_list[0]['intent']
    confidence = float(intents_list[0]['probability'])

    # Check if this is a location-specific query
    location_specific_intents = ['population_query', 'area_query', 'household_query', 'rural_urban_query',
                                 'villages_towns_query']

    if intent_tag in location_specific_intents:
        location = extract_location_from_query(user_input)

        if location and location in location_data:
            # Check if we have valid data
            data = location_data[location]

            if data.get('population', 0) == 0 and data.get('area', 0) == 0:
                return f"I found {location} in the database, but it appears to have no recorded data."

            return format_demographic_response(intent_tag, location, data)
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

        # For comparison queries, provide top populated regions
        top_regions = sorted(location_data.items(), key=lambda x: x[1].get('population', 0), reverse=True)[:5]
        response = "Here are the top 5 most populated regions:\n"
        for i, (region, data) in enumerate(top_regions, 1):
            pop = data.get('population', 0)
            if pop > 0:
                response += f"{i}. {region}: {pop:,.0f} people\n"
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
        ints = predict_class(text)
        res = get_response(ints, intents, text)
        return res
    except Exception as e:
        return "Sorry, I encountered an error processing your request. Please try again."


def process_query(text):
    """Process and enhance user queries for better understanding"""
    if not text:
        return ""

    text = text.lower().strip()

    # Replace common synonyms
    replacements = {
        'how many people': 'population',
        'how big': 'area',
        'size of': 'area of',
        'families': 'households',
        'rural urban': 'rural-urban ratio',
        'number of people': 'population',
        'how much area': 'area'
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# Test function for debugging
def test_chatbot():
    """Test function to check if chatbot is working"""
    test_queries = [
        "what is the population of goa",
        "area of delhi",
        "households in mumbai"
    ]

    print("Testing chatbot responses:")
    for query in test_queries:
        processed = process_query(query)
        response = chatbot_response(processed)
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

    print("Indian Demographics Chatbot")
    print("Ask me about population, area, households, villages, or towns in different parts of India!")
    print("(Type 'quit' to stop)\n")

    while True:
        message = input("You: ")
        if message.lower() in ['quit', 'exit', 'bye', 'goodbye', 'stop']:
            print("Bot: Thank you for using the Indian Demographics Chatbot! Goodbye! 👋")
            break

        processed_message = process_query(message)
        response = chatbot_response(processed_message)
        print("Bot:", response)
        print()