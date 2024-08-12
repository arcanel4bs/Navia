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

if not GOOGLE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("API keys not set. Please check your environment variables.")

# Initialize Gemini & Maps
gmaps = GoogleMaps(key=GOOGLE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

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

@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html', google_api_key=GOOGLE_API_KEY)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    app.logger.info('Processing user input request...')
    
    user = User.query.filter_by(session_id=session['user_id']).first()
    if not user:
        user = User(session_id=session['user_id'])
        db.session.add(user)
        db.session.commit()

    chat_history = ChatHistory.query.filter_by(user_id=user.id).order_by(ChatHistory.timestamp.desc()).limit(10).all()
    chat_history = [{"role": ch.role, "parts": [ch.message]} for ch in reversed(chat_history)]
    
    chat = model.start_chat(enable_automatic_function_calling=True, history=chat_history)
    
    response = chat.send_message(user_input)
    
    directions_info = None
    for part in response.parts:
        if part.function_call and part.function_call.name == "get_detailed_directions":
            args = part.function_call.args
            directions_info = get_detailed_directions(args['origin'], args['destination'])
            if "error" not in directions_info:
                directions_summary = f"Here are the directions from {directions_info['origin']} to {directions_info['destination']}:\n"
                directions_summary += f"Distance: {directions_info['distance']}\n"
                directions_summary += f"Duration: {directions_info['duration']}\n"
                directions_summary += "Steps:\n"
                for i, step in enumerate(directions_info['steps'], 1):
                    directions_summary += f"{i}. {step['instruction']} ({step['distance']} - {step['duration']})\n"
                
                response = chat.send_message(directions_summary)
            else:
                response = chat.send_message(f"I'm sorry, but I couldn't retrieve directions: {directions_info['error']}")
            break
    
    response_text = ' '.join(part.text for part in response.parts if part.text)
    app.logger.info('Generated response')
    
    # Save chat history to database
    db.session.add(ChatHistory(user_id=user.id, message=user_input, role="user"))
    db.session.add(ChatHistory(user_id=user.id, message=response_text, role="model"))
    db.session.commit()
    
    return jsonify({
        "response": response_text,
        "directions_info": directions_info
    })

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8080)
