import os
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
from googlemaps import Client as GoogleMaps
from dotenv import load_dotenv
import urllib.parse
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import uuid
from google.oauth2 import service_account
from google.cloud import speech_v2, texttospeech
import base64
import json
import re
import sys

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '3d6f45a5fc12445dbac2f59c3b6c7cb1')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///navia.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Set API Keys
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GOOGLE_CLOUD_PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT_ID')

if not GOOGLE_API_KEY or not GEMINI_API_KEY or not GOOGLE_CLOUD_PROJECT_ID:
    raise ValueError("API keys not set. Please check your environment variables.")

# Initialize Gemini & Maps
gmaps = GoogleMaps(key=GOOGLE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Replace 'sa_speech.json' with the actual path to your service account key file


credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

credentials = service_account.Credentials.from_service_account_file(credentials_path)

speech_client = speech_v2.SpeechClient(credentials=credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

if not app.debug:
    file_handler = RotatingFileHandler('navia.log', maxBytes=10240, backupCount=10, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    # Also add a StreamHandler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    app.logger.addHandler(console_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info('Navia startup')

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('chat_history', lazy=True))

# Set up logging
if not app.debug:
    file_handler = RotatingFileHandler('navia.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Navia startup')

# Define direction parameters & instructions
def get_detailed_directions(origin: str, destination: str) -> dict:
    try:
        directions = gmaps.directions(origin, destination, mode="driving")
        app.logger.info('Fetching directions...')

        if directions:
            route = directions[0]
            leg = route['legs'][0]

            map_url = f"https://maps.googleapis.com/maps/api/staticmap?size=600x300&maptype=roadmap"
            map_url += f"&markers=color:green|label:A|{origin}"
            map_url += f"&markers=color:red|label:B|{destination}"
            map_url += f"&path=enc:{route['overview_polyline']['points']}"
            map_url += f"&key={GOOGLE_API_KEY}"
            app.logger.info('Map URL generated')

            steps = [
                {
                    "instruction": step['html_instructions'],
                    "distance": step['distance']['text'],
                    "duration": step['duration']['text'],
                    "start_location": step['start_location'],
                    "end_location": step['end_location']
                }
                for step in leg['steps']
            ]
            
            origin_coords = f"{leg['start_location']['lat']},{leg['start_location']['lng']}"
            dest_coords = f"{leg['end_location']['lat']},{leg['end_location']['lng']}"
            waze_url = f"https://www.waze.com/ul?navigate=yes&from={urllib.parse.quote(origin_coords)}&to={urllib.parse.quote(dest_coords)}"
            
            return {
                "origin": leg['start_address'],
                "destination": leg['end_address'],
                "distance": leg['distance']['text'],
                "duration": leg['duration']['text'],
                "map_url": map_url,
                "start_location": leg['start_location'],
                "end_location": leg['end_location'],
                "steps": steps,
                "waze_url": waze_url,
                "overview_polyline": route['overview_polyline']['points']
            }
        else:
            return {"error": "No route found"}
    except Exception as e:
        app.logger.error(f'Error in get_detailed_directions: {str(e)}')
        return {"error": str(e)}

# Set the Model & Function calling structure directions parameters
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash-latest',
    tools=[
        {
            "function_declarations": [
                {
                    "name": "get_detailed_directions",
                    "description": "Get detailed directions between two locations, always provide distance and time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string", "description": "Starting point of the route"},
                            "destination": {"type": "string", "description": "End point of the route"}
                        },
                        "required": ["origin", "destination"]
                    }
                }
            ]
        }
    ]
)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_content = audio_file.read()

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="long",
    )

    request_stt = speech_v2.RecognizeRequest(
        recognizer=f"projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        content=audio_content,
    )

    response = speech_client.recognize(request=request_stt)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return jsonify({"transcript": transcript})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-C",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return jsonify({"audio_content": base64.b64encode(response.audio_content).decode('utf-8')})


SYSTEM_PROMPT = """
You are Navia, an AI navigation assistant designed to help users plan their trips and answer questions about travel. Your primary function is to provide accurate and helpful information about routes, travel times, and distances between locations. Here are your key capabilities and guidelines:

1. Route Information: You can provide detailed directions, including distance and travel time, between any two locations. Always use the get_detailed_directions function to ensure accuracy.

2. Travel Planning: Help users plan their trips by suggesting optimal routes, estimating travel times, and providing information about potential stops along the way.

3. Local Knowledge: Utilize your internal knowledge to provide information about points of interest, landmarks, and general facts about locations along the route.

4. Time and Distance Calculations: When asked about travel times or distances, always refer to the data provided by the get_detailed_directions function. This ensures the most up-to-date and accurate information.

5. Alternative Transportation: While your primary focus is on driving directions, you can suggest alternative modes of transportation when appropriate, such as public transit, walking, or cycling for shorter distances.

6. Safety and Comfort: Provide tips for safe and comfortable travel, such as suggesting rest stops on long journeys or advising about potential traffic conditions.

7. Consistency: Always strive to provide consistent information. If you've given specific details about a route, refer back to that information in subsequent responses.

8. Clarification: If a user's query is ambiguous or lacks necessary details, try to answer as helpfully as possible befire asking for clarification to ensure you provide the most relevant and accurate information.

9. Limitations: Be honest about your limitations. If you're unsure about something or if the information requested is beyond your capabilities, communicate this clearly to the user.

10. Engaging Responses: While maintaining professionalism, try to make your responses engaging and conversational. You can express enthusiasm about interesting destinations or routes.

Remember, your primary goal is to assist users in navigating and understanding their travel routes effectively. Always prioritize accuracy and helpfulness in your responses.

11. Do not generate asterisks like this **Title** in any situation whatsoever.

12. Do not use emojis!

13. Be straight forward in your responses
"""


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        app.logger.info(f'Received user input: {user_input}')
        
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        app.logger.info(f'User ID: {session["user_id"]}')
        
        user = User.query.filter_by(session_id=session['user_id']).first()
        if not user:
            user = User(session_id=session['user_id'])
            db.session.add(user)
            db.session.commit()
            app.logger.info(f'Created new user with ID: {user.id}')
        
        chat_history = ChatHistory.query.filter_by(user_id=user.id).order_by(ChatHistory.timestamp.desc()).limit(10).all()
        chat_history = [{"role": ch.role, "parts": [ch.message]} for ch in reversed(chat_history)]
        app.logger.info(f'Retrieved chat history: {chat_history}')
        
        chat = model.start_chat(history=chat_history)
        app.logger.info('Started chat with Gemini model')
        
        # Include the system prompt in the conversation
        system_message = chat.send_message(SYSTEM_PROMPT)
        app.logger.info('Sent system prompt to Gemini')
        
        response = chat.send_message(user_input)
        app.logger.info(f'Received response from Gemini: {response}')
        
        directions_info = None
        for part in response.parts:
            if part.function_call and part.function_call.name == "get_detailed_directions":
                args = part.function_call.args
                directions_info = get_detailed_directions(args['origin'], args['destination'])
                app.logger.info(f'Retrieved directions info: {directions_info}')
                if "error" not in directions_info:
                    directions_summary = f"Here are the directions from {directions_info['origin']} to {directions_info['destination']}:\n"
                    directions_summary += f"Distance: {directions_info['distance']}\n"
                    directions_summary += f"Duration: {directions_info['duration']}\n"
                    directions_summary += "Steps:\n"
                    for i, step in enumerate(directions_info['steps'], 1):
                        directions_summary += f"{i}. {step['instruction']} ({step['distance']} - {step['duration']})\n"
                    
                    response = chat.send_message(directions_summary)
                    app.logger.info(f'Sent directions summary to Gemini: {directions_summary}')
                else:
                    response = chat.send_message(f"I'm sorry, but I couldn't retrieve directions: {directions_info['error']}")
                    app.logger.error(f'Error retrieving directions: {directions_info["error"]}')
                break
        
        response_text = ' '.join(part.text for part in response.parts if part.text)
        
        # Remove asterisks from the response
        response_text = re.sub(r'\*+', '', response_text)
        
        app.logger.info(f'Final response text (asterisks removed): {response_text}')
        
        # Save chat history to database
        db.session.add(ChatHistory(user_id=user.id, message=user_input, role="user"))
        db.session.add(ChatHistory(user_id=user.id, message=response_text, role="model"))
        db.session.commit()
        app.logger.info('Saved chat history to database')
        
        # Generate audio for the response
        audio_content = None
        if response_text:
            tts_response = text_to_speech_internal(response_text)
            audio_content = tts_response.get('audio_content')
            app.logger.info('Generated audio content for response')
        
        app.logger.info(f'Returning response with directions_info: {directions_info is not None}')
        return jsonify({
            "response": response_text,
            "directions_info": directions_info,
            "audio_content": audio_content
        })
    
    except Exception as e:
        app.logger.error(f'Error in chat route: {str(e)}')
        return jsonify({"error": str(e)}), 500

def text_to_speech_internal(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-C",
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return {"audio_content": base64.b64encode(response.audio_content).decode('utf-8')}

@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html', google_api_key=GOOGLE_API_KEY)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8080)
