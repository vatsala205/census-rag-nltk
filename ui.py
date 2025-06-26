from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sys
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for chatbot functions
chatbot_response = None
process_query = None


def load_chatbot():
    """Load chatbot functions with better error handling"""
    global chatbot_response, process_query

    try:
        # Import your existing chatbot code
        from census_chat import chatbot_response as cr, process_query as pq
        chatbot_response = cr
        process_query = pq
        print("✅ Chatbot loaded successfully!")
        return True
    except ImportError as e:
        print(f"❌ Error importing chatbot: {e}")
        print("Make sure census_chat.py and all required files are in the same directory:")
        print("- intents.json")
        print("- location_data.json")
        print("- training_data.pkl")
        print("- chatbot_model.h5")
        return False
    except Exception as e:
        print(f"❌ Unexpected error loading chatbot: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


# Load chatbot when app starts
chatbot_loaded = load_chatbot()

# Store chat sessions (in production, use a proper database)
chat_sessions = {}


@app.route('/')
def index():
    """Serve the frontend HTML"""
    status_message = "✅ Backend and chatbot loaded successfully!" if chatbot_loaded else "❌ Chatbot failed to load"
    status_color = "green" if chatbot_loaded else "red"

    return f"""
    <html>
    <head>
        <title>Indian Demographics Chatbot</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .status {{ color: {status_color}; font-weight: bold; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            #messageInput {{ width: 300px; padding: 10px; margin-right: 10px; }}
            button {{ padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            #response {{ margin-top: 20px; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; min-height: 50px; }}
            .error {{ background: #f8d7da; border-left-color: #dc3545; color: #721c24; }}
            .example {{ margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Indian Demographics Chatbot Backend</h1>
            <p class="status">{status_message}</p>

            <h3>Test the Chatbot API:</h3>
            <form id="testForm">
                <input type="text" id="messageInput" placeholder="Enter your message" value="what is the area of goa">
                <button type="submit">Send</button>
            </form>
            <div id="response">Response will appear here...</div>

            <h3>Example Queries:</h3>
            <div class="example">
                <strong>Try these examples:</strong><br>
                • "what is the population of delhi"<br>
                • "area of goa"<br>
                • "households in mumbai"<br>
                • "male population of kerala"<br>
                • "villages in rajasthan"
            </div>

            <h3>API Endpoints:</h3>
            <ul>
                <li><code>POST /chat</code> - Main chat endpoint</li>
                <li><code>GET /health</code> - Health check</li>
                <li><code>GET /test</code> - Test chatbot functionality</li>
            </ul>
        </div>

        <script>
        document.getElementById('testForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            const message = document.getElementById('messageInput').value;
            const responseDiv = document.getElementById('response');

            // Show loading
            responseDiv.innerHTML = '🤖 Processing...';
            responseDiv.className = '';

            try {{
                const response = await fetch('/chat', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{message: message}})
                }});

                const data = await response.json();

                if (data.error) {{
                    responseDiv.innerHTML = '<strong>Error:</strong> ' + data.error;
                    responseDiv.className = 'error';
                }} else {{
                    responseDiv.innerHTML = '<strong>🤖 Bot:</strong> ' + data.response;
                    responseDiv.className = '';
                }}
            }} catch (error) {{
                responseDiv.innerHTML = '<strong>Error:</strong> ' + error.message;
                responseDiv.className = 'error';
            }}
        }});

        // Add click handlers for examples
        document.querySelectorAll('.example').forEach(example => {{
            example.addEventListener('click', (e) => {{
                if (e.target.textContent.includes('•')) {{
                    const text = e.target.textContent.replace('• ', '').replace('"', '').replace('"', '');
                    document.getElementById('messageInput').value = text;
                }}
            }});
        }});
        </script>
    </body>
    </html>
    """


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with better error handling"""
    try:
        if not chatbot_loaded:
            return jsonify({
                'error': 'Chatbot not loaded properly',
                'message': 'Please check server logs for details'
            }), 500

        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                'error': 'No message provided',
                'message': 'Please provide a message in the request body'
            }), 400

        user_message = data['message'].strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({
                'error': 'Empty message',
                'message': 'Please provide a non-empty message'
            }), 400

        print(f"📝 Received message: '{user_message}'")

        # Process the query using your existing function
        processed_message = process_query(user_message)
        print(f"🔄 Processed message: '{processed_message}'")

        # Get response from your chatbot
        bot_response = chatbot_response(processed_message)
        print(f"🤖 Bot response: '{bot_response}'")

        # Store conversation in session (optional)
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        chat_sessions[session_id].append({
            'user': user_message,
            'bot': bot_response,
            'processed': processed_message
        })

        return jsonify({
            'response': bot_response,
            'session_id': session_id,
            'status': 'success',
            'processed_query': processed_message
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error in chat endpoint: {e}")
        print(f"📋 Full traceback: {error_details}")

        return jsonify({
            'error': 'Internal server error',
            'message': 'Sorry, I encountered an error processing your request.',
            'details': str(e) if app.debug else None
        }), 500


@app.route('/test', methods=['GET'])
def test_chatbot():
    """Test endpoint to verify chatbot functionality"""
    if not chatbot_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Chatbot not loaded'
        })

    test_queries = [
        "what is the population of goa",
        "area of delhi",
        "households in mumbai"
    ]

    results = []
    for query in test_queries:
        try:
            processed = process_query(query)
            response = chatbot_response(processed)
            results.append({
                'query': query,
                'processed': processed,
                'response': response,
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'query': query,
                'error': str(e),
                'status': 'error'
            })

    return jsonify({
        'test_results': results,
        'chatbot_loaded': chatbot_loaded
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if chatbot_loaded else 'degraded',
        'message': 'Indian Demographics Chatbot API is running',
        'chatbot_loaded': chatbot_loaded
    })


@app.route('/sessions', methods=['GET'])
def get_sessions():
    """Get chat sessions (for debugging)"""
    return jsonify({
        'sessions': chat_sessions,
        'session_count': len(chat_sessions)
    })


@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear a specific chat session"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')

    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return jsonify({'message': f'Session {session_id} cleared'})
    else:
        return jsonify({'message': 'Session not found'}), 404


@app.route('/reload', methods=['POST'])
def reload_chatbot():
    """Reload the chatbot (useful for development)"""
    global chatbot_loaded

    # Force reload of modules
    if 'census_chat' in sys.modules:
        del sys.modules['census_chat']

    chatbot_loaded = load_chatbot()

    return jsonify({
        'status': 'success' if chatbot_loaded else 'error',
        'message': 'Chatbot reloaded successfully' if chatbot_loaded else 'Failed to reload chatbot',
        'chatbot_loaded': chatbot_loaded
    })


if __name__ == '__main__':
    print("🚀 Starting Indian Demographics Chatbot Server...")

    if not chatbot_loaded:
        print("⚠️  WARNING: Chatbot not loaded properly. Server will run but chat functionality may not work.")

    print("📡 Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)