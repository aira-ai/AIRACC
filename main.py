import os
import random
from datetime import datetime, timedelta
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template, flash
from flask_sqlalchemy import SQLAlchemy
from flask_paginate import Pagination, get_page_parameter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sentence_transformers import SentenceTransformer,util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dlib

from werkzeug.security import generate_password_hash, check_password_hash
import cv2

from flask import url_for
from datetime import timedelta
import torch
from sqlalchemy import func
import numpy as np
import re
from flask import session
import pandas as pd
import plotly.express as px
from flask import redirect, url_for
import openai
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from geopy.geocoders import Nominatim 
from langdetect import detect
import nltk
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import spacy
nlp = spacy.load("en_core_web_sm")

from collections import defaultdict

nltk.download('stopwords')
nltk.download('punkt')



# Initialize Flask app and database
app = Flask(__name__)
app.secret_key="asdfaksjdhfajsdhfjkashdfjkashdfjkashdfjkhajsdfkasd"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///incident_reports.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder for media uploads
app.config['UPLOAD_FOLDER_POI'] = 'static/photos/POI'  # Folder for media uploads
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}  # Allowed file types for uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db = SQLAlchemy(app)
CORS(app)
# Load OpenAI API key from environment variable for security
openai.api_key = "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"


# Initialize geolocator
geolocator = Nominatim(user_agent="incident_dashboard")

class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    gps_lat_min = db.Column(db.Float, nullable=True)
    gps_lat_max = db.Column(db.Float, nullable=True)
    gps_long_min = db.Column(db.Float, nullable=True)
    gps_long_max = db.Column(db.Float, nullable=True)
    viewers = db.Column(db.Text, nullable=True)  # List of user IDs allowed to view
    comments = db.relationship('Comment', backref='announcement', lazy=True)

    def __repr__(self):
        return f'<Announcement {self.title}>'

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, nullable=False)  # Assuming users have unique IDs
    announcement_id = db.Column(db.Integer, db.ForeignKey('announcement.id'), nullable=False)

    def __repr__(self):
        return f'<Comment {self.content[:20]}>'
    
# USERS model
class USERS(db.Model):
    __tablename__ = 'USERS'
    user_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    access = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    mobile = db.Column(db.String(15), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    location_id = db.Column(db.String(50), nullable=True)
    instance_id = db.Column(db.String(50), nullable=True)
    photo = db.Column(db.String(200), nullable=True)
    role =  db.Column(db.String(20), nullable=True)

class Response(db.Model):
    __tablename__ = 'responses'
    response_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident.id'), nullable=False)
    tag = db.Column(db.String(20), nullable=False)

class CitizenData(db.Model):
    __tablename__ = "citizendata"
    
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Primary key
    ADDRESS = db.Column(db.Text, nullable=True)  # Address can be NULL
    PRECINCT = db.Column(db.Text, nullable=True)  # Precinct can be NULL
    NAME = db.Column(db.Text, nullable=False)  # Name is required
    GENDER = db.Column(db.String(10), nullable=True)  # Gender as a short string
    BIRTHDAY = db.Column(db.String(10), nullable=True)  # Birthday in text format (e.g., "YYYY-MM-DD")
    BARANGAY = db.Column(db.Text, nullable=True)  # Barangay can be NULL
    longitude = db.Column(db.Text, nullable=True)  # Barangay can be NULL
    latitude = db.Column(db.Text, nullable=True)  # Barangay can be NULL
    countrycode = db.Column(db.Text, nullable=True)  # Barangay can be NULL
    def __repr__(self):
        return f"<CitizenData(ID={self.ID}, NAME='{self.NAME}', ADDRESS='{self.ADDRESS}', PRECINCT='{self.PRECINCT}', GENDER='{self.GENDER}', BIRTHDAY='{self.BIRTHDAY}', BARANGAY='{self.BARANGAY}',longitude='{self.longitude}',latitude='{self.latitude}', BARANGAY='{self.countrycode}')>"

    # Define the relationship with the Incident model
    #incident = db.relationship('Incident', backref='incident_responses', lazy=True)

    def to_dict(self):
        return {
            "response_id": self.response_id,
            "user_id": self.user_id,
            "response": self.response,
            "timestamp": self.timestamp,
            "incident_id": self.incident_id,
            "tag": self.tag
        }
    
class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(10), nullable=False)

    questions = db.relationship('Question', backref='survey', lazy=True)

class Question(db.Model):
    __tablename__ = 'questions'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)  # The survey question text
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp of question creation
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'), nullable=False)
    input_method = db.Column(db.String(100), nullable=False)
    # Relationship to responses
    responses = db.relationship('QResponses', backref='question', lazy=True)

    def __repr__(self):
        return f"<Question id={self.id} text={self.text}>"


class QResponses(db.Model):
    __tablename__ = 'QResponses'
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)
    response_text = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sentiment = db.Column(db.Text, nullable=True)
    action = db.Column(db.Text, nullable=True)
    longitude = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Text, nullable=True)
    location = db.Column(db.Text, nullable=True)
    name = db.Column(db.Text, nullable=True)
    address = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<QResponses id={self.id} question_id={self.question_id} user_id={self.user_id}>"

    def to_dict(self):
        return {
            'id': self.id,
            'question_id': self.question_id,
            'user_id': self.user_id,
            'response_text': self.response_text,
            'language': self.language,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'sentiment': self.sentiment or 'neutral',  # Ensure non-null value
            'action': self.action,
            'name': self.name,
            'address': self.address,
            'longitude': self.longitude or '0',  # Default to '0' if None
            'latitude': self.latitude or '0',  # Default to '0' if None
            'location': self.location or '',  # Default to empty string if Non       # Assuming it exists for 'similarity' mode
            
        }


class IncidentAnalysis(db.Model):
    __tablename__ = 'incident_analysis'
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident.id'), nullable=False)  # Ensure foreign key is correct
    action_points = db.Column(db.String, nullable=True)
    report_text = db.Column(db.Text, nullable=False)
    tokens = db.Column(db.String, nullable=True)
    user_id = db.Column(db.Integer, nullable=False)
    

    incident = db.relationship('Incident', backref='analyses')  # Relationship with the Incident model

    
    def to_dict(self):
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "action_points": self.action_points,
            "report_text": self.report_text,
            "tokens": self.tokens,
            "user_id": self.user_id,
            
        }

class PersonOfInterest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    alias = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    last_seen_location = db.Column(db.String(200), nullable=True)
    last_seen_date = db.Column(db.DateTime, nullable=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident.id'), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)  # New field for photo path
    user_id = db.Column(db.Integer, nullable=False)
    
    incident = db.relationship('Incident', backref='persons_of_interest')
# Define the Incident model
# Incident Model
class Incident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caller_name = db.Column(db.String(100))
    contact_number = db.Column(db.String(15))
    report_text = db.Column(db.Text)
    media_path = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    tokens = db.Column(db.Text, nullable=True)
    openai_analysis = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(200), nullable=True)
    language = db.Column(db.String(20), nullable=True)
    tag = db.Column(db.String(20), nullable=True)
    actionpoints = db.Column(db.Text)
    notes = db.Column(db.Text)
    type =  db.Column(db.String(20), nullable=True)
    assigned_authorities = db.Column(db.Text, nullable=True)  # New field for assigned authorities
    user_id = db.Column(db.Integer, nullable=False)
    disregard_words = db.Column(db.JSON, default=[])
    complainant =  db.Column(db.String(100), nullable=True)
    defendant =  db.Column(db.String(100), nullable=True)

    #responses = db.relationship('Response', backref='incident_ref', lazy=True)


    def to_dict(self):
        """Convert the Incident object to a serializable dictionary."""
        return {
            'id': self.id,
            'report_text': self.report_text,
            'caller_name':self.caller_name,
            'contact_number':self.contact_number,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'tag': self.tag,
            'category': self.category,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'tokens': self.tokens,
            'openai_analysis': self.openai_analysis,
            'location': self.location,
            'actionpoints': self.actionpoints,
            'notes': self.notes,
            'type': self.type,
            'assigned_authorities': self.assigned_authorities,
            'user_id': self.user_id,
            'disregard_words': self.disregard_words,
            'complainant': self.complainant,
            'defendant':self.defendant,
            'media_path': self.media_path if self.media_path else None

        }
    def set_tokens(self):
        """Populate the tokens field based on report_text."""
        if self.report_text:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(self.report_text)
            self.tokens = " ".join([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]])  # Only using nouns and proper nouns as tokens

    def __repr__(self):
        return f'<Incident {self.id} - {self.category}>'
    
    
    
class Message(db.Model):
    __tablename__ = 'messages'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('USERS.user_id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('USERS.user_id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='sent')  # "sent", "delivered", "read"
    message_type = db.Column(db.String(50), default='text')  # Can be 'text', 'image', 'file', etc.
    is_read = db.Column(db.Boolean, default=False)
    reply_to = db.Column(db.Integer, db.ForeignKey('messages.id'))  # Foreign key to the parent message, if replying
    attachment_url = db.Column(db.String(255))  # URL or path to an attachment

    # Relationship to the User table (sender and receiver)
    sender = db.relationship('USERS', foreign_keys=[sender_id])
    receiver = db.relationship('USERS', foreign_keys=[receiver_id])

    # Relationship to itself for replies
    parent_message = db.relationship('Message', remote_side=[id])  # For replies to a message

    def __repr__(self):
        return f"<Message {self.id} from {self.sender_id} to {self.receiver_id}>"

# Initialize database
with app.app_context():
    db.create_all()

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper functions for analysis
def fetch_incidents(filters):
    # Build dynamic query using filters
    query = Incident.query.filter(*filters)  # Filters are passed as a list of conditions
    return query.all()

import pandas as pd

def analyze_incidents(incidents):
    # Process the incidents to generate stats, handling None timestamps
    data = []
    for incident in incidents:
        if incident.timestamp is not None:
            date = incident.timestamp.date()
        else:
            date = None  # Use None or a default date, e.g., datetime(1970, 1, 1).date()
        data.append((incident.id, incident.location, incident.category, date))
    
    df = pd.DataFrame(data, columns=["id", "location", "category", "date"])

    # Analyze stats by location, category, and date
    location_stats = df.groupby('location').size().reset_index(name='incident_count')
    category_stats = df.groupby('category').size().reset_index(name='incident_count')
    date_stats = df.groupby('date').size().reset_index(name='incident_count')

    return location_stats, category_stats, date_stats





def process_predictions(predictions):
    for prediction in predictions:
        # Debugging: print forecasted incidents
        forecasted_incidents = prediction.get('incident_prediction', {}).get('forecasted_incidents_next_2_weeks', [])
        print(f"Forecasted Incidents: {forecasted_incidents}")  # Add this line to see the structure of the data

        # Remove duplicate action points
        action_points = list(set(prediction.get('incident_prediction', {}).get('action_points', [])))
        prediction['incident_prediction']['action_points'] = action_points

        # Add focus categories and incidents
        focus_data = defaultdict(int)
        
        if forecasted_incidents:
            for idx, forecast_value in enumerate(forecasted_incidents):
                # Ensure that we have both category and count data
                if isinstance(forecast_value, dict):  # Ensure that forecast is a dictionary
                    category = forecast_value.get('category')
                    count = forecast_value.get('count', 0)
                    
                    if category and isinstance(count, (int, float)):
                        focus_data[category] += count
        
        # Sort categories by count for focus
        sorted_focus = sorted(focus_data.items(), key=lambda x: x[1], reverse=True)
        prediction['focus_categories'] = sorted_focus[:3]  # Top 3 categories to focus on

        print(f"Focus Categories: {prediction['focus_categories']}")  # Add this line to debug focus categories

    return predictions

def predict_incidents(incidents):
    print(f"Incidents sample: {incidents[:5]}")
    
    # Convert incidents to DataFrame for easier manipulation
    df = pd.DataFrame([(incident.location, incident.timestamp, incident.category) 
                       for incident in incidents], columns=["location", "timestamp", "category"])

    # Preprocessing and Feature Engineering
    df['day'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year

    # 1. Calculate frequency of incidents in each location
    incident_frequency = df.groupby('location').size().reset_index(name='incident_count')

    # 2. Detect relationships between incidents using clustering (DBSCAN)
    location_category_data = df[['location', 'category']].drop_duplicates()
    location_category_data['location'] = location_category_data['location'].astype('category').cat.codes
    location_category_data['category'] = location_category_data['category'].astype('category').cat.codes
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(location_category_data)

    # DBSCAN clustering to detect patterns
    db = DBSCAN(eps=0.5, min_samples=3).fit(scaled_data)
    location_category_data['cluster'] = db.labels_

    # 3. Predict incidents in the next 2 weeks using time series forecasting (Exponential Smoothing)
    df['date'] = pd.to_datetime(df['timestamp'])
    incidents_by_date = df.groupby('date').size().reset_index(name='incident_count')
    incidents_by_date.set_index('date', inplace=True)

    # Check if there are at least 2 data points
    if len(incidents_by_date) < 2:
        return "Insufficient data to predict incidents. At least two data points are required."

    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(incidents_by_date['incident_count'], trend='add', seasonal=None)
    forecast = model.fit().forecast(steps=14)  # Predict for the next 14 days

    # 4. Detailed Interpretation of Forecast
    def interpret_forecast(forecast, threshold=2):
        """
        Interpret the forecast to identify trends.
    
        :param forecast: Pandas Series of forecasted values.
        :param threshold: The minimum change to consider a trend significant.
        :return: A description of the trend.
        """
        if len(forecast) > 0:
            # Calculate the overall change in forecast
            forecast_change = forecast.iloc[-1] - forecast.iloc[0]

            # Determine the trend based on the change and threshold
            if abs(forecast_change) < threshold:
                forecast_trend = "stable"
            elif forecast_change > 0:
                forecast_trend = "increasing"
            else:
                forecast_trend = "decreasing"

            # Determine the peak day if the trend is increasing
            peak_day = None
            if forecast_trend == "increasing":
                peak_day = forecast.idxmax()

            # Generate trend description
            if forecast_trend == "increasing":
                trend_description = (
                    f"Incident frequency is predicted to rise steadily over the next two weeks. "
                    f"The highest surge in incidents is expected around {peak_day.strftime('%Y-%m-%d') if peak_day else 'a peak date'}. "
                    f"Recommended actions: Monitor activity closely and allocate resources accordingly."
                )
            elif forecast_trend == "decreasing":
                trend_description = (
                    "Incident frequency is expected to decrease steadily over the next two weeks. "
                    "This suggests a decline in activity. Maintain regular monitoring to ensure continued stability."
                )
            else:
                trend_description = (
                    "Incident frequency is expected to remain stable over the next two weeks. "
                    "No significant changes in activity are anticipated."
                )
        else:
            trend_description = "Insufficient data to predict incident trends."

        return trend_description

    forecast_interpretation = interpret_forecast(forecast, threshold=5)

    # 5. Action points based on the prediction
    action_points = []
    for location, count in incident_frequency.values:
        if count > 10:
            action_points.append(f"Allocate additional patrols to {location}.")
        category_data = df[df['location'] == location].groupby('category').size().reset_index(name='incident_count')
        frequent_categories = category_data[category_data['incident_count'] > 5]
        for category in frequent_categories['category']:
            action_points.append(f"Organize community outreach for {category} in {location}.")

    # 6. Define start_date and end_date for the prediction period
    start_date = datetime.today().date()  # Current date as start date
    end_date = start_date + timedelta(days=14)  # 14 days ahead for end date

    # 7. Return prediction data in a structured format
    prediction_data = []
    for location, incident_count in incident_frequency.values:
        most_common_category = df[df['location'] == location]['category'].mode().iloc[0]  # Mode of the category
        prediction_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # Current time as the prediction time

        prediction = {
            'location': location,
            'incident_count': incident_count,
            'most_common_category': most_common_category,
            'prediction_time': prediction_time,
            'start_date': start_date.strftime('%Y-%m-%d'),  # Add start_date to prediction
            'end_date': end_date.strftime('%Y-%m-%d'),      # Add end_date to prediction
            'incident_prediction': {
                'forecasted_incidents_next_2_weeks': forecast.tolist(),
                'forecast_interpretation': forecast_interpretation,
                'action_points': action_points
            },
            'prediction': f"High incident frequency predicted for {location}. Recommended actions: {', '.join(action_points)}"
        }
        prediction_data.append(prediction)

    # Process the predictions to add focus categories and remove duplicates
    prediction_data = process_predictions(prediction_data)

    print(f"Prediction Data: {prediction_data}")
    return prediction_data



def handle_natural_language_query(query):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes crime, missing persons, sanitation, public works, health or food security incident data. Be concise and brief only. "},
            {"role": "user", "content": f"Analyze and answer this query based on incidents data: {query}"}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Function to categorize crime text using AI
def categorize_incident(report_text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(report_text)
    filtered_text = " ".join([w for w in words if not w.lower() in stop_words])

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes various incidents such as crime, missing persons,sanitation, public works, health or food security."},
            {"role": "user", "content": f"Classify this report: {filtered_text} based on your knowledge. Do not be verbose, just use one or two words for the category."}
        ],
        max_tokens=10,
    )
    return response['choices'][0]['message']['content'].strip()

# Function to analyze report using OpenAI
def analyze_report(report_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a crime lab assistant that analyzes incident reports and provides action points based on complex algorithms and detective investigation manuals."},
            {"role": "user", "content": f"Analyze the following report and provide brief action points: {report_text}"}
        ],
        max_tokens=150,
    )
    return response['choices'][0]['message']['content'].strip()

# Function to get location from latitude and longitude
def get_location(latitude, longitude):
    try:
        location = geolocator.reverse((latitude, longitude), language='en', timeout=30)
        return location.address if location else "Unknown location"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Home route - render the report form
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/add_announcement', methods=['GET', 'POST'])
def add_announcement():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        gps_lat_min = request.form.get('gps_lat_min')
        gps_lat_max = request.form.get('gps_lat_max')
        gps_long_min = request.form.get('gps_long_min')
        gps_long_max = request.form.get('gps_long_max')
        viewers = request.form['viewers']  # Comma-separated list of user IDs
        
        new_announcement = Announcement(
            title=title,
            content=content,
            gps_lat_min=gps_lat_min,
            gps_lat_max=gps_lat_max,
            gps_long_min=gps_long_min,
            gps_long_max=gps_long_max,
            viewers=viewers
        )
        
        db.session.add(new_announcement)
        db.session.commit()
        return redirect(url_for('index'))
    
    return render_template('add_announcement.html')


@app.route('/edit_announcement/<int:id>', methods=['GET', 'POST'])
def edit_announcement(id):
    announcement = Announcement.query.get_or_404(id)
    
    if request.method == 'POST':
        announcement.title = request.form['title']
        announcement.content = request.form['content']
        announcement.gps_lat_min = request.form.get('gps_lat_min')
        announcement.gps_lat_max = request.form.get('gps_lat_max')
        announcement.gps_long_min = request.form.get('gps_long_min')
        announcement.gps_long_max = request.form.get('gps_long_max')
        announcement.viewers = request.form['viewers']
        
        db.session.commit()
        return redirect(url_for('index'))
    
    return render_template('edit_announcement.html', announcement=announcement)

@app.route('/delete_announcement/<int:id>', methods=['GET'])
def delete_announcement(id):
    announcement = Announcement.query.get_or_404(id)
    db.session.delete(announcement)
    db.session.commit()
    return redirect(url_for('announcements'))

@app.route('/announcements')
def index():
    announcements = Announcement.query.all()
    return render_template('announcements.html', announcements=announcements)

@app.route('/add_comment/<int:announcement_id>', methods=['POST'])
def add_comment(announcement_id):
    comment_content = request.form['comment']
    new_comment = Comment(content=comment_content, user_id=1, announcement_id=announcement_id)  # Replace 1 with the logged-in user's ID
    db.session.add(new_comment)
    db.session.commit()
    return redirect(url_for('index'))

from difflib import SequenceMatcher
import json



# Function to prepare data for rendering (e.g., in the heatmap)
def prepare_data(data, mode, color_map):
    sanitized_data = []

    for item in data:
        filter_value = item.get('filter_value', 'unknown')
        color_code = color_map[mode].get(filter_value, "gray")
        
        if mode == "sentiment":
            sentiment = item.get('sentiment', 'neutral').lower()
            if 'positive' in sentiment:
                filter_value = 'positive'
            elif 'negative' in sentiment:
                filter_value = 'negative'
            else:
                filter_value = 'neutral'
            color_code = color_map['sentiment'].get(filter_value, "gray")
        elif mode == "similarity":
            group = item.get('group', 'unknown')
            filter_value = group  # Use the group name as the filter value
            color_code = color_map['similarity'].get(group, "gray")  # Get color based on group

        sanitized_data.append({
            'latitude': float(item.get('latitude', 0)),
            'longitude': float(item.get('longitude', 0)),
            'response_text': item.get('response_text', ''),
            'location': item.get('location', ''),
            'filter_value': filter_value,
            'color_code': color_code  # Ensure color_code is passed for similarity
        })

    return sanitized_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import openai

from googletrans import Translator

class IncidentService:
    def __init__(self, search_terms, stop_words=None):
        self.search_terms = search_terms
        self.stop_words = stop_words or []
        self.translator = Translator()

    def _clean_search_terms(self):
        """Remove custom stop words from the search terms."""
        return [term for term in self.search_terms if term.lower() not in self.stop_words]

    def _get_translations(self, terms):
        """Translate terms into English and back to the original language."""
        translations = []
        for term in terms:
        # Translate term to English
            translated = self.translator.translate(term, src='auto', dest='en').text
        
        # Detect the language of the term and translate back to the original language
            original_language = self.translator.detect(term).lang  # Detect the original language
            back_translated = self.translator.translate(translated, src='en', dest=original_language).text
        
            translations.extend([term, translated, back_translated])
        return list(set(translations))

    def _calculate_similarity(self, texts):
        """Calculate similarity scores between search terms and texts."""
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts + [' '.join(self.search_terms)])
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        return similarity_scores

    def search_incidents(self):
        """Search the Incident table based on search terms."""
        incidents = Incident.query.all()
        texts = [incident.report_text or "" for incident in incidents]
        self.search_terms = self._clean_search_terms()
        multilingual_terms = self._get_translations(self.search_terms)

        similarity_scores = self._calculate_similarity(texts)
        sorted_indices = similarity_scores.argsort()[::-1][:20]  # Top 20 results
        return [incidents[i] for i in sorted_indices]

    def search_persons(self):
        """Search the PersonOfInterest table based on search terms."""
        persons = PersonOfInterest.query.all()
        texts = [person.description or "" for person in persons]
        self.search_terms = self._clean_search_terms()
        multilingual_terms = self._get_translations(self.search_terms)

        similarity_scores = self._calculate_similarity(texts)
        sorted_indices = similarity_scores.argsort()[::-1][:5]  # Top 5 results
        return [persons[i] for i in sorted_indices]

    def search_citizens(self):
        """Search the CitizenData table based on search terms."""
        citizens = CitizenData.query.all()
        texts = [citizen.ADDRESS or "" for citizen in citizens]
        self.search_terms = self._clean_search_terms()
        multilingual_terms = self._get_translations(self.search_terms)

        similarity_scores = self._calculate_similarity(texts)
        sorted_indices = similarity_scores.argsort()[::-1][:5]  # Top 5 results
        return [citizens[i] for i in sorted_indices]
    
    

@app.route('/NLsearch', methods=['GET', 'POST'])
def nlsearch():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template('nlsearch.html', error="Please enter a search query.")
        
        # Translate non-English queries to English using GPT-4
        try:
            translated_query = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Translate the following query to English."},
                    {"role": "user", "content": query}
                ]
            )['choices'][0]['message']['content'].strip()
        except Exception as e:
            return render_template('nlsearch.html', error=f"Translation failed: {e}")

        # Perform NLP for keywords and entity extraction
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(translated_query)

            # Extract and prioritize keywords
            keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            entities = [ent.text for ent in doc.ents]
            search_terms = list(set(keywords + entities))
        except Exception as e:
            return render_template('nlsearch.html', error=f"NLP processing failed: {e}")

        # Include the original query for multilingual support
        search_terms.append(query)

        # Define custom stop words
        stop_words = [
            "incident", "report", "find", "of", "data", "the", "a", "an", "i want to",
            "barangay", "want", "i", "to", "ko", "gusto", "maghanap", "ka", "ng"
        ]

        # Instantiate IncidentService and perform searches
        try:
            service = IncidentService(search_terms, stop_words=stop_words)
            incident_results = service.search_incidents()
            person_results = service.search_persons()
            citizen_results = service.search_citizens()
        except Exception as e:
            return render_template('nlsearch.html', error=f"Search failed: {e}")

        # Render the results
        return render_template(
            'nlsearch.html',
            query=query,
            translated_query=translated_query,
            incidents=incident_results,
            persons=person_results,
            citizens=citizen_results
        )

    # For GET requests, simply render the search page
    return render_template('nlsearch.html')




# Route for the heatmap page
@app.route('/heatmap', methods=['GET', 'POST'])
def heatmap():
    question_id = request.form.get('question_id', 1)
    mode = request.form.get('mode', 'sentiment')

    questions = Question.query.all()
    responses = QResponses.query.filter_by(question_id=question_id).all()

    # Define color map for sentiment and similarity modes
    color_map = {
        "sentiment": {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow"
        },
        "similarity": {
            "group1": "blue", "group2": "green", "group3": "purple",
            "group4": "orange", "group5": "pink", "group6": "green",
            "group7": "yellow", "group8": "red", "group9": "brown", "group10": "grey"
        }
    }

    # Handle the 'similarity' mode
    if mode == 'similarity':
        grouped_responses = group_responses_by_similarity(responses)
        responses_data = grouped_responses
    else:
        # Ensure we are working with proper dictionary objects
        responses_data = [response.to_dict() if hasattr(response, 'to_dict') else response for response in responses]

    sanitized_data = prepare_data(responses_data, mode, color_map)

    return render_template('heatmap.html',
                           heatmap_data=sanitized_data,
                           questions=questions,
                           mode=mode,
                           question_id=question_id,
                           filtered_responses=sanitized_data,
                           color_map=color_map)

# Function to group responses based on similarity (using text)
def group_responses_by_similarity(responses):
    grouped = defaultdict(list)
    color_palette = [
        "blue", "green", "purple", "orange", "pink", 
        "green", "yellow", "red", "brown", "grey"
    ]

    for response in responses:
        normalized_text = response.response_text.strip().lower()
        grouped[normalized_text].append(response)

    grouped_responses = []
    for idx, (group_name, group) in enumerate(grouped.items()):
        group_color = color_palette[idx % len(color_palette)]  # Assign a color from the palette
        for response in group:
            grouped_responses.append({
                'latitude': response.latitude,
                'longitude': response.longitude,
                'response_text': response.response_text,
                'location': response.location,
                'group': f'group{idx+1}',  # Assign group name like 'group1', 'group2', etc.
                'color_code': group_color  # Pass the assigned color
            })

    return grouped_responses

# Route to fetch all barangays
@app.route('/get_barangays', methods=['GET'])
def get_barangays():
    barangays = db.session.query(CitizenData.BARANGAY).distinct().all()
    barangay_list = [barangay[0] for barangay in barangays if barangay[0] is not None]  # Exclude NULL values
    return jsonify(barangay_list)


@app.route('/update_coordinates', methods=['POST'])
def update_coordinates():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        
        # Check if all required fields are present
        if not data or 'id' not in data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({'success': False, 'error': 'Invalid request. Missing id, latitude, or longitude.'}), 400
        
        # Extract data
        citizen_id = data['id']
        latitude = data['latitude']
        longitude = data['longitude']

        # Validate latitude and longitude
        if not isinstance(latitude, (float, int)) or not isinstance(longitude, (float, int)):
            return jsonify({'success': False, 'error': 'Latitude and longitude must be numbers.'}), 400
        
        # Fetch the citizen record from the database
        citizen = CitizenData.query.get(citizen_id)
        if not citizen:
            return jsonify({'success': False, 'error': f'Citizen with ID {citizen_id} not found.'}), 404
        
        # Update the citizen's coordinates
        citizen.latitude = latitude
        citizen.longitude = longitude
        
        # Commit the changes to the database
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Coordinates for citizen ID {citizen_id} updated successfully.',
            'updated_data': {
                'id': citizen.ID,
                'name': citizen.NAME,
                'latitude': citizen.latitude,
                'longitude': citizen.longitude,
            }
        }), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': 'Database error occurred.', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': 'An unexpected error occurred.', 'details': str(e)}), 500
    
@app.route('/citizen_dashboard', methods=['GET'])
def citizen_dashboard():
    # Pagination parameters
    page = request.args.get('page', 1, type=int)  # Default to page 1
    per_page = 100  # Show 100 citizens per page

    # Search parameters from query string
    search_params = {
        "barangay": request.args.get('barangay'),
        "address": request.args.get('address'),
        "gender": request.args.get('gender'),
        "precinct": request.args.get('precinct'),
        "location": request.args.get('location'),
        "birthday": request.args.get('birthday'),
        "name": request.args.get('name')
    }

    # Construct the filter query
    filters = []
    if search_params['barangay']:
        filters.append(CitizenData.BARANGAY.ilike(f"%{search_params['barangay']}%"))
    if search_params['address']:
        filters.append(CitizenData.ADDRESS.ilike(f"%{search_params['address']}%"))
    if search_params['gender']:
        filters.append(CitizenData.GENDER == search_params['gender'])
    if search_params['precinct']:
        filters.append(CitizenData.precinct.ilike(f"%{search_params['precinct']}%"))
    if search_params['location']:
        filters.append(or_(
            CitizenData.latitude.ilike(f"%{search_params['location']}%"),
            CitizenData.longitude.ilike(f"%{search_params['location']}%")
        ))
    if search_params['birthday']:
        filters.append(CitizenData.BIRTHDAY == search_params['birthday'])
    if search_params['name']:
        filters.append(CitizenData.NAME.ilike(f"%{search_params['name']}%"))

    # Query database with filters and pagination
    query = CitizenData.query
    if filters:
        query = query.filter(and_(*filters))

    citizens = query.order_by(CitizenData.ID).paginate(page=page, per_page=per_page)

    return render_template('citizen_dashboard.html', citizens=citizens, search_params=search_params)

@app.route('/add_citizen', methods=['GET', 'POST'])
def add_citizen():
    if request.method == 'POST':
        data = request.form
        new_citizen = CitizenData(
            NAME=data['NAME'],
            ADDRESS=data['ADDRESS'],
            BARANGAY=data['BARANGAY'],
            PRECINCT=data.get('PRECINCT'),
            GENDER=data.get('GENDER'),
            BIRTHDAY=data.get('BIRTHDAY'),
            longitude=data.get('longitude'),
            latitude=data.get('latitude'),
            countrycode=data.get('countrycode')
        )
        db.session.add(new_citizen)
        db.session.commit()
        return redirect(url_for('citizen_dashboard'))
    return render_template('add_citizen.html')

@app.route('/edit_citizen/<int:id>', methods=['GET', 'POST'])
def edit_citizen(id):
    # Fetch the citizen by ID or return a 404 error if not found
    citizen = CitizenData.query.get_or_404(id)
    
    if request.method == 'POST':
        # Fetch form data from the request
        data = request.form
        
        # Update citizen fields with data from the form
        try:
            citizen.NAME = data.get('NAME', '').strip()  # Ensure name is not None
            citizen.ADDRESS = data.get('ADDRESS', '').strip()
            citizen.BARANGAY = data.get('BARANGAY', '').strip()
            citizen.PRECINCT = data.get('PRECINCT', '').strip()  # Optional field
            citizen.GENDER = data.get('GENDER', '').strip()
            citizen.BIRTHDAY = data.get('BIRTHDAY', '').strip()
            citizen.longitude = data.get('longitude', None)  # Optional field
            citizen.latitude = data.get('latitude', None)  # Optional field
            citizen.countrycode = data.get('countrycode', '').strip()  # Optional field
            
            # Commit changes to the database
            db.session.commit()
            
            # Redirect to the citizen dashboard after successful update
            return redirect(url_for('citizen_dashboard'))
        except Exception as e:
            # Handle any errors during the update process
            db.session.rollback()
            flash(f"An error occurred while updating the citizen: {e}", "error")
            return redirect(url_for('edit_citizen', id=id))
    
    # Render the edit citizen form
    return render_template('edit_citizen.html', citizen=citizen)
from sqlalchemy import and_

@app.route('/search_citizens', methods=['GET'])
def search_citizens():
    name = request.args.get('name', '')
    address = request.args.get('address', '')
    results = CitizenData.query.filter(
        CitizenData.NAME.like(f'%{name}%'),
        CitizenData.ADDRESS.like(f'%{address}%')
    ).all()
    return jsonify([{
        'ID': c.ID,
        'NAME': c.NAME,
        'ADDRESS': c.ADDRESS,
        'BARANGAY': c.BARANGAY,
        'latitude': c.latitude,
        'longitude': c.longitude
    } for c in results])

@app.route('/get_citizens', methods=['GET'])
def get_citizens():
    try:
        # Extract query parameters
        barangay = request.args.get('barangay', '')
        name = request.args.get('name', '')
        address = request.args.get('address', '')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 30))

        print(f"Received request with parameters: Barangay={barangay}, Name={name}, Address={address}, Page={page}, Limit={limit}")
        
        # Build query
        query = CitizenData.query
        if barangay:
            query = query.filter(CitizenData.BARANGAY == barangay)
        if name:
            query = query.filter(CitizenData.NAME.like(f'%{name}%'))
        if address:
            query = query.filter(CitizenData.ADDRESS.like(f'%{address}%'))

        # Correct pagination
        citizens = query.paginate(page=page, per_page=limit, error_out=False)
        
        print(f"Fetched {len(citizens.items)} citizens from the database.")

        # Prepare response
        response_data = {
            'records': [{
                'ID': citizen.ID,
                'name': citizen.NAME,
                'address': citizen.ADDRESS,
                'barangay': citizen.BARANGAY,
                'latitude': citizen.latitude,   # Ensure 'latitude' is lowercase
                'longitude': citizen.longitude, # Ensure 'longitude' is lowercase
            } for citizen in citizens.items],
            'totalRecords': citizens.total
        }

        return jsonify(response_data)
    except Exception as e:
        # Log the exception with detailed info
        print(f"Error fetching citizen data: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Print the full traceback to the console
        return jsonify({"error": "Failed to fetch citizen data"}), 500

# Route to render the citizen map page
@app.route('/citizen_map')
def map_citizens():
    return render_template('citizen_map.html')

@app.route("/answer_survey", methods=["GET", "POST"])
def answer_survey():
    if request.method == "POST":
        survey_id = request.form['survey_id']
        latitude = request.form.get('latitude')  # Get latitude from the form
        longitude = request.form.get('longitude')  # Get longitude from the form
        name= request.form.get('name')
        address = request.form.get('address')
        # Collect answers and process them
        answers = {}
        for question in Question.query.filter_by(survey_id=survey_id).all():
            answer_text = request.form.get(f"answer_{question.id}")
            answers[question.id] = answer_text

        # Save responses to the database
        for question_id, answer_text in answers.items():
            response = QResponses(
                question_id=question_id,
                response_text=answer_text,
                name=name,
                address=address,
                timestamp=datetime.utcnow(),
                latitude=latitude,  # Save latitude with the response
                longitude=longitude  # Save longitude with the response
            )
            db.session.add(response)

        db.session.commit()
        return redirect(url_for('survey_submitted'))

    surveys = Survey.query.filter_by(status="Active").all()
    return render_template('answer_survey.html', surveys=surveys)

@app.route("/survey_submitted")
def survey_submitted():
    return render_template('survey_submitted.html')




@app.route('/create_survey', methods=['GET', 'POST'])
def create_survey():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        status = request.form['status']
        new_survey = Survey(title=title, description=description, status=status)
        db.session.add(new_survey)
        db.session.commit()
        return redirect('/create_survey')
    # Fetch all surveys to display in the template
    surveys = Survey.query.all()
    return render_template('create_survey.html', surveys=surveys)

@app.route('/edit_survey/<int:survey_id>', methods=['GET', 'POST'])
def edit_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    if request.method == 'POST':
        survey.title = request.form['title']
        survey.description = request.form['description']
        survey.status = request.form['status']
        db.session.commit()
        return redirect('/create_survey')
    return render_template('edit_survey.html', survey=survey)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()



@app.route('/view_responses')
def view_responses():
    surveys = Survey.query.all()
    with db.session.no_autoflush:  # Disable autoflush during processing
        for survey in surveys:
            survey.questions = Question.query.filter_by(survey_id=survey.id).all()
            for question in survey.questions:
                question.responses = QResponses.query.filter_by(question_id=question.id).all()
                for response in question.responses:
                    if not response.sentiment or not response.action:
                        sentiment_result = compute_sentiment(response.response_text)
                        response.sentiment = f"{sentiment_result['label']} ({sentiment_result['score']})"
                        
                        # Adjust action to ensure it gets a proper value, not a tuple
                        action_result = generate_action_points2(response.response_text)
                        
                        # Extract status code or message from the response if it's a tuple
                        if isinstance(action_result, tuple):
                            response.action = action_result[0]  # Take the first element (e.g., '200 OK')
                        else:
                            response.action = str(action_result)  # Or any default logic to convert to string
                        
                        print(f"Formatted Sentiment: {response.sentiment}")
                        print(f"Action: {response.action}")
                        db.session.add(response)  # Stage changes for commit
        db.session.commit()  # Commit changes to the database
    return render_template('view_responses.html', surveys=surveys)

def compute_sentiment(text):
    """Compute sentiment using VADER and TextBlob."""
    vader_result = vader_analyzer.polarity_scores(text)
    vader_score = vader_result["compound"]

    blob = TextBlob(text)
    blob_score = blob.sentiment.polarity

    # Aggregate sentiment scores
    aggregated_score = vader_score
    aggregated_label = classify_aggregated_sentiment(aggregated_score)

    return {
        "label": aggregated_label,
        "score": round(vader_score, 2),  # Only include VADER score for this purpose
    }

def generate_action_points2(text):
    """Generate action points using OpenAI."""
    try:
        api_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a government assistant generating action points for user feedback."},
                {"role": "user", "content": f"Based on this feedback: '{text}', provide recommended action points in a concise format."}
            ]
        )
        action_points = api_response["choices"][0]["message"]["content"]
        print (action_points)
        return action_points.strip()  # Return the content as a string
    except Exception as e:
        print(f"Error generating action points: {e}")
        return "No action points available."  # Return a string even on error

def classify_aggregated_sentiment(aggregated_score):
    """Classify the sentiment based on the aggregated sentiment score."""
    positive_threshold = 0.6
    negative_threshold = 0.4

    if aggregated_score >= positive_threshold:
        return "Positive"
    elif aggregated_score <= negative_threshold:
        return "Negative"
    else:
        return "Neutral"

# Load spaCy's English model
@app.template_filter('datetimeformat')
def datetimeformat(value):
    if isinstance(value, datetime):  # Now you can directly use datetime
        return value.strftime('%Y-%m-%d %H:%M:%S')  # Adjust the format as needed
    return value


# Load BERT model and tokenizer
# Initialize BERT tokenizer and model for embedding generation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
# Assuming get_bert_embeddings is replaced by a sentence transformer model or similar
def get_bert_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Better for semantic search
    return model.encode(text)

@app.route('/manage_questions', methods=['GET'])
def manage_questions():
    # Fetch all surveys with questions
    surveys = Survey.query.all()  # Ensure surveys are fetched
    if not surveys:
        print("No surveys found!")  # Debugging message
    for survey in surveys:
        survey.questions = Question.query.filter_by(survey_id=survey.id).all()
    return render_template('manage_questions.html', surveys=surveys)

    return render_template('manage_questions.html', surveys=surveys)

@app.route('/add_question', methods=['POST'])
def add_question():
    survey_id = request.form['survey_id']
    question_text = request.form['text']
    input_method = request.form['input_method']
    
    # Save the question to the database with the selected input_method
    new_question = Question(
        survey_id=survey_id,
        text=question_text,
        input_method=input_method
    )
    db.session.add(new_question)
    db.session.commit()
    
    return redirect('/manage_questions')

@app.route('/edit_question/<int:question_id>', methods=['GET', 'POST'])
def edit_question(question_id):
    question = Question.query.get_or_404(question_id)
    if request.method == 'POST':
        question.text = request.form.get('text')
        db.session.commit()
        return redirect(url_for('manage_questions'))
    return render_template('edit_question.html', question=question)

@app.route('/delete_question/<int:question_id>', methods=['GET'])
def delete_question(question_id):
    question = Question.query.get_or_404(question_id)
    db.session.delete(question)
    db.session.commit()
    return redirect(url_for('manage_questions'))

@app.route('/toggle_survey_status/<int:survey_id>', methods=['POST'])
def toggle_survey_status(survey_id):
    survey = Survey.query.get(survey_id)
    if survey:
        survey.status = 'Inactive' if survey.status == 'Active' else 'Active'
        db.session.commit()
    return redirect('/create_survey')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        access = "active"

        # Hash the password before saving
        hashed_password = generate_password_hash(password)

        new_user = USERS(
            first_name=first_name,
            last_name=last_name,
            email=email,
            mobile=mobile,
            username=username,
            password=hashed_password,
            role=role
        )

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('User added successfully!', 'success')
            return redirect(url_for('manage_users'))  # Redirect to a page listing all users
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')

    return render_template('manage_users.html')  # render the page with the form

@app.route('/update_role/<int:user_id>', methods=['POST'])
def update_role(user_id):
    user = db.session.get(USERS, user_id)
    if user:
        user.role = request.form['role']
        db.session.commit()
        flash('User role updated successfully!', 'success')
    else:
        flash('User not found.', 'danger')
    return redirect(url_for('manage_users'))

import sqlite3  # Add this import at the top of your file
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@app.route('/generate_action_points/<int:incident_id>', methods=['GET'])
def generate_action_points(incident_id):
    # Fetch the related incident
    incident = Incident.query.get(incident_id)
    if not incident:
        return jsonify({"success": False, "error": "Incident not found."}), 404

    if 'user_id' not in session:
        return jsonify({"success": False, "error": "User not authenticated."}), 401

    report_text = incident.report_text

    try:
        # Call OpenAI GPT-4 API to generate action points
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant generating actionable steps for incident reports. Use Philippine setting. Be brief and direct. Add authorities that need to be contacted."},
                {"role": "user", "content": f"Generate action points for this incident report: {report_text}"}
            ]
        )
        
        try:
            action_points = response['choices'][0]['message']['content']
            tokens_used = response.get('usage', {}).get('total_tokens', 0)
        except (KeyError, IndexError) as e:
            return jsonify({"success": False, "error": f"Invalid response from OpenAI: {str(e)}"}), 500

        # Ensure values are properly extracted and not tuples
        action_points = str(action_points)  # Convert action points to string
        tokens_used = int(tokens_used)     # Ensure tokens is an integer
        report_text = str(report_text)     # Ensure report text is a string
        user_id = int(session['user_id'])  # Ensure user_id is an integer

        # Create a new IncidentAnalysis entry
        analysis = IncidentAnalysis(
            incident_id=incident_id,           # Should be an integer, not a tuple
            report_text=report_text,           # Should be a string
            action_points=action_points,       # Should be a string
            tokens=tokens_used,                # Should be an integer
            user_id=user_id                    # Should be an integer
        )
        db.session.add(analysis)
        db.session.commit()

        return redirect(request.referrer)

    except openai.error.OpenAIError as e:
        return jsonify({"success": False, "error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/send_message/<int:receiver_id>', methods=['POST'])
def send_message(receiver_id):
    try:
        request_data = request.get_json()
        print(f"Received data: {request_data}")

        sender_id = session['user_id']
        message_text = request_data.get('message')

        if not sender_id or not message_text:
            print("Error: Missing sender_id or message")
            return jsonify({"error": "Sender ID and message are required"}), 400

        # Create new message object
        new_message = Message(sender_id=sender_id, receiver_id=receiver_id, message=message_text)
        print(f"Created message object: {new_message}")

        # Add message object to the session and commit
        db.session.add(new_message)

        # Retry mechanism for database insert
        retries = 3
        for _ in range(retries):
            try:
                db.session.commit()
                print(f"Message saved with id: {new_message.id}")
                
                # Return a success response with message details
                return jsonify({
                    "success": True,
                    "message": "Message sent successfully",
                    "data": {
                        "sender_id": new_message.sender_id,
                        "receiver_id": new_message.receiver_id,
                        "message": new_message.message,
                        "id": new_message.id
                    }
                }), 200
            except sqlite3.OperationalError as e:
                print(f"Error: {e}")
                time.sleep(1)
        
        # If commit fails, return an error
        return jsonify({"error": "Database is locked. Please try again later."}), 500

    except Exception as e:
        print(f"Error sending message: {e}")
        return jsonify({"error": "Failed to send message", "details": str(e)}), 500
       

@app.route('/users', methods=['GET', 'POST'])
def manage_users():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 10
    search_query = request.args.get('search', '')

    if search_query:
        users_query = USERS.query.filter(
            (USERS.first_name.ilike(f'%{search_query}%')) |
            (USERS.last_name.ilike(f'%{search_query}%')) |
            (USERS.email.ilike(f'%{search_query}%'))
        )
    else:
        users_query = USERS.query

    # Paginate results using SQLAlchemy's built-in paginate method
    paginated_users = users_query.paginate(page=page, per_page=per_page, error_out=False)

    users = paginated_users.items  # Extract items for the current page
    pagination = Pagination(page=page, total=paginated_users.total, search=bool(search_query), per_page=per_page, css_framework='bootstrap4')

    return render_template('manage_users.html', users=users, pagination=pagination, search_query=search_query)

# Route to activate/deactivate user
@app.route('/toggle_user/<int:user_id>', methods=['POST'])
def toggle_user(user_id):
    user = USERS.query.get(user_id)
    if not user:
        flash('User not found!', 'danger')
        return redirect(url_for('manage_users'))

    # Toggle access field
    user.access = 'active' if user.access != 'active' else 'inactive'
    db.session.commit()
    flash(f"User {user.first_name} {user.last_name}'s status changed to {user.access}.", 'success')
    return redirect(url_for('manage_users'))

@app.route('/get_messages/<int:user_id>', methods=['GET'])
def get_messages(user_id):
    try:
        messages = Message.query.filter(
            (Message.sender_id == user_id) | (Message.receiver_id == user_id)
        ).all()

        response_messages = []
        for msg in messages:
            sender = USERS.query.get(msg.sender_id)
            receiver = USERS.query.get(msg.receiver_id)
            
            response_messages.append({
                'sender': sender.username if sender else 'Unknown',
                'sender_name': sender.username if sender else 'Unknown',
                'text': msg.message
            })
        
        return jsonify({"messages": response_messages}), 200

    except Exception as e:
        print(f"Error fetching messages: {e}")
        return jsonify({"error": "Failed to fetch messages", "details": str(e)}), 500

@app.route('/persons_of_interest')
def persons_of_interest():
    persons = PersonOfInterest.query.all()
    return render_template('persons_of_interest.html', persons=persons)

@app.route('/add_person', methods=['POST'])
def add_person():
    name = request.form['name']
    alias = request.form.get('alias')
    description = request.form.get('description')
    last_seen_location = request.form.get('last_seen_location')
    user_id = session['user_id']
    
    notes = request.form.get('notes')

     # Handle file upload
    photo = request.files['photo']
    photo_path = None
    if photo and allowed_file(photo.filename):
        filename = secure_filename(photo.filename)
        photo_path = os.path.join(app.config['UPLOAD_FOLDER_POI'], filename)
        photo.save(photo_path)

    last_seen_date_str = request.form.get('last_seen_date')  # e.g., "2024-11-21T14:30"

# Convert to a datetime object
    if last_seen_date_str:
        last_seen_date = datetime.strptime(last_seen_date_str, "%Y-%m-%dT%H:%M")
        print(f"Converted datetime: {last_seen_date}")
    else:
        print("No date provided")
        
    
    new_person = PersonOfInterest(
        name=name,
        alias=alias,
        description=description,
        last_seen_location=last_seen_location,
        last_seen_date=last_seen_date,
        notes=notes,
        photo_path = photo_path,
        user_id=user_id
    )
    db.session.add(new_person)
    db.session.commit()
    return redirect(url_for('persons_of_interest'))

# Route for registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        mobile = request.form['mobile']
        access = 'inactive'  # Default access level for new users

        if USERS.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))

        if USERS.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = USERS(
            first_name=first_name,
            last_name=last_name,
            email=email,
            username=username,
            password=password,
            mobile=mobile,
            access=access,
            role="REPORTER"
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful as reporter! Please wait for activation before logging in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


# Route for login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Query the user from the database
        user = USERS.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            # Successful login
            session['user_id'] = user.user_id
            session['username'] = user.username
            session['access'] = user.access
            session['role'] = user.role

            if user.role == 'ADMIN':
                return redirect(url_for('dashboard'))
            elif user.role == 'REPORTER':
                return render_template("index.html")  # Redirect to reporter's home page
        else:
            # Invalid credentials or unauthorized role
            flash('Invalid username, password, or unauthorized access.', 'danger')
    
    # Render login page for GET requests or after a failed login
    return render_template('login.html')

@app.route('/session', methods=['GET'])
def get_session():
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'user_id': session['user_id'],
            'username': session['username'],
            'role': session['role']
        })
    return jsonify({'logged_in': False})

# Route for logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Route to render the analysis form
@app.route('/analyze', methods=['GET', 'POST'])
def analyze_data():
    if request.method == 'POST':
        # Get user inputs
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        location = request.form.get('location')

        # Query the database for incidents within the date range and location
        incidents = Incident.query.filter(
            Incident.timestamp >= datetime.strptime(start_date, '%Y-%m-%d'),
            Incident.timestamp <= datetime.strptime(end_date, '%Y-%m-%d'),
            Incident.location == location
        ).all()

        # Aggregate data for analysis
        reports = [incident.report_text for incident in incidents]
        combined_reports = " ".join(reports)
        data = combined_reports
        print(f"Data being sent to OpenAI: {data}")
        # Call OpenAI   for intelligent analysis
        openai_response = analyze_with_openai(combined_reports)

        return render_template('analysis_result.html', incidents=incidents, analysis=openai_response)

    return render_template('analysis_form.html')

def analyze_with_openai(data):
    """
    Use OpenAI to generate an intelligent analysis of the provided data.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent data analyst. Provide detailed analysis and actionable insights based on the user's input data."
                },
                {
                    "role": "user",
                    "content": f"Provide a detailed analysis and actionable insights for the following data:\n\n{data}"
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error in OpenAI analysis: {e}"


@app.route('/analyze_incident/<int:incident_id>', methods=['GET', 'POST'])
def analyze_incident(incident_id):
    
    # Fetch the incident object from the database
    incident = Incident.query.get_or_404(incident_id)
    
    # Check if the request method is POST (form submission)
    if request.method == 'POST':
        # Extract action points and what to do from the form data
        action_points = request.form.get('action_points', '')
        
        
        # Extract tokens from the report_text (can be used for comparison)
        report_text = incident.report_text
        tokens = incident.tokens  # Tokens already exist in the incident object
        
        # Save the new analysis to the database
        analysis = IncidentAnalysis(
            incident_id=incident.id,
            action_points=action_points,
            report_text = report_text,
            tokens=tokens  # Tokens as comma-separated string
        )

        db.session.add(analysis)
        db.session.commit()

        # Now suggest action points based on similar incidents
    

    # Handle GET request (for displaying incident details and any suggested points)
    return render_template('analyze_incident.html', incident=incident)




@app.route('/dispatch', methods=['GET', 'POST'])
def dispatch():
    if request.method == 'POST':
        caller_name = request.form['caller_name']
        contact_number = request.form['contact_number']
        report_text = request.form['report_text']
        location = request.form['location']
        tag = request.form['tag']
        notes = request.form['notes']
        type ="dispatch"
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        complainant = request.form.get('complainant')
        defendant = request.form.get('defendant')

        category = categorize_incident(report_text)
        
    
        # Tokenization and analysis are performed here
        tokens = ", ".join([w for w in word_tokenize(report_text) if w.lower() not in stopwords.words('english')])
    
    # Detect the language of the report
        language = detect(report_text)

        # AI Analysis
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert dispatcher AI assistant."},
                {"role": "user", "content": f"Analyze this incident and suggest appropriate authorities or departments to dispatch: {report_text}"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        suggestions = openai_response['choices'][0]['message']['content'].strip()

        # Save incident to database
        new_incident = Incident(
            caller_name=caller_name,
            tag = tag,
            category=category,
            tokens=tokens,
            language = language,
            notes=notes,
            type = type,
            contact_number=contact_number,
            report_text=report_text,
            location=location,
            openai_analysis=suggestions,
            latitude = latitude,
            longitude=longitude,
            complainant = complainant,
            defendant = defendant,
            user_id = session['user_id'],
            timestamp=datetime.utcnow()
        )
        db.session.add(new_incident)
        db.session.commit()

        return redirect(url_for('dispatch'))

    incidents = Incident.query.order_by(Incident.timestamp.desc()).all()
    return render_template('dispatch.html', incidents=incidents)

@app.route('/send_dispatch/<int:incident_id>', methods=['POST'])
def send_dispatch(incident_id):
    incident = Incident.query.get_or_404(incident_id)
    authorities = request.form['assigned_authorities']
    incident.assigned_authorities = authorities
    db.session.commit()
    return jsonify({'message': 'Dispatch sent successfully!'})

@app.route('/ai', methods=['GET', 'POST'])
def ai_page():
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    # Default filter: incidents from today and yesterday
    filter_conditions = []
    
    # Collecting search query from the form (location, category, and query)
    location_filter = request.form.get('location')
    category_filter = request.form.get('category')
    query_filter = request.form.get('query')

    group_by = request.args.get('group_by', 'tokens')  # Default to 'tokens'
    page = request.args.get('page', default=1, type=int)
    total_pages = 10

    # Building the filter conditions based on the provided filters
    if request.method == 'POST':
        if location_filter:
            filter_conditions.append(Incident.location == location_filter)
        if category_filter:
            filter_conditions.append(Incident.category == category_filter)
    
    # Fetch incidents based on filter conditions
    incidents = fetch_incidents(filter_conditions)

    if not incidents:
        return render_template('ai.html', 
                               location_stats=[], 
                               category_stats=[], 
                               crime_prediction=[],
                               location_fig_html='',
                               category_fig_html='',
                               group_by = group_by,
                               page=page,
                               total_pages= total_pages,
                               query_result='No incidents found for the selected filters.')

    # Analyze the incidents to generate stats
    location_stats, category_stats, date_stats = analyze_incidents(incidents)
    
    # Create visualizations for incidents by location and category
    fig_location = px.bar(location_stats, x='location', y='incident_count', title="Incidents by Location")
    fig_category = px.bar(category_stats, x='category', y='incident_count', title="Incidents by Category")
    
    # Generate HTML for the charts
    location_fig_html = fig_location.to_html(full_html=False)
    category_fig_html = fig_category.to_html(full_html=False)
    
    # Predict incidents based on the filtered incidents
    incident_prediction = predict_incidents(incidents)
    session['incident_prediction'] = incident_prediction

    # Handle the natural language query if provided
    query_result = ''
    if query_filter:
        query_result = handle_natural_language_query(query_filter)

    # Render the template with data and charts
    return render_template('ai.html', 
                           location_stats=location_stats.to_dict(orient='records'),
                           category_stats=category_stats.to_dict(orient='records'),
                           incident_prediction=incident_prediction,
                           location_fig_html=location_fig_html, 
                           category_fig_html=category_fig_html,
                           group_by=group_by,
                           page=page,
                           total_pages = total_pages,
                           query_result=query_result)



@app.route('/update_tag/<incident_id>', methods=['POST'])
def update_tag(incident_id):
    tag = request.form['tag']
    incident = Incident.query.get(incident_id)
    if incident:
        incident.tag = tag
        db.session.commit()

    return redirect(url_for('incident_details', incident_id=incident_id))

@app.route('/add_response/<int:incident_id>', methods=['POST'])
def add_response(incident_id):
    user_id = session['user_id']  # Replace with actual user ID from session or authentication
    response_text = request.form.get('response')
    timestamp = datetime.now()
    tag = request.form['tag']

    new_response = Response(
        incident_id=incident_id,
        user_id=user_id,
        response=response_text,
        timestamp=timestamp,
        tag=tag
    )
    db.session.add(new_response)
    db.session.commit()

    incident = Incident.query.get(incident_id)
    if incident:
        incident.tag = tag  # Update the incident's tag
        db.session.commit()

    return redirect(url_for('incident_details', incident_id=incident_id))

@app.route('/map')
def map_page():
    return render_template('map.html')  # This will render the map.html template


@app.route('/map_data', methods=['GET'])
def map_data():
    # Get the 'start_date', 'end_date' and 'search_query' query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    search_query = request.args.get('query', '').lower()  # New parameter for search query
    print(search_query)
    # Default to today's date if no date range is provided
    if not start_date or not end_date:
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = end_date = today

    # Convert string dates to datetime objects (with default time)
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Set the time for start_date to 00:00:00 and end_date to 23:59:59
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    # Query incidents from the database and filter by date range
    incidents = Incident.query.filter(Incident.timestamp >= start_date, Incident.timestamp <= end_date)

    if search_query:
        # If there's a natural language search query, filter based on report_text or category
        doc = nlp(search_query)
        
        # Extract keywords (nouns and proper nouns) and named entities from the query
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        entities = [ent.text for ent in doc.ents]
        
        # Combine keywords and entities for search criteria (remove duplicates)
        search_terms = set(keywords + entities)
        
        # Remove stopwords from search terms
        stop_words = set(stopwords.words('english'))
        filtered_search_terms = [term for term in search_terms if term.lower() not in stop_words]

        # Query the database with the filtered search terms using TF-IDF and Cosine Similarity
        if filtered_search_terms:
            # Extract report texts from all incidents
            all_incidents = Incident.query.all()
            all_reports = [i.report_text for i in all_incidents]

            # Vectorize the report text and the search query
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_reports + [' '.join(filtered_search_terms)])
            
            # Compute cosine similarity between the query and the incident reports
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            
            # Sort incidents by similarity to the search query
            similar_indices = cosine_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar incidents
            incidents = [all_incidents[i] for i in similar_indices]

            print(f"THERE ARE INCIDENTS: {incidents}")

    # Convert the query results to a list of incident details
    filtered_incidents = [
        {
            'id': incident.id,
            'latitude': incident.latitude,
            'longitude': incident.longitude,
            'category': incident.category,
            'location': incident.location,
            'report_text': incident.report_text,
            'timestamp': incident.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        for incident in incidents
    ]

    app.logger.debug(f"Filtered Map Data from {start_date} to {end_date} with search query '{search_query}': {filtered_incidents}")

    return jsonify(filtered_incidents)






def group_related_incidents(incidents):
    """Groups incidents by token, category, and report_text similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    grouped = {}

    for incident in incidents:
        # Create a string key for grouping
        key = f"{incident.tokens}-{incident.category}"
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(incident)

    # Further group by text similarity within each key
    for key, incidents_list in grouped.items():
        vectorizer = TfidfVectorizer()
        texts = [incident.report_text for incident in incidents_list]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Add advanced grouping logic if needed here

    return grouped



def generate_insights(grouped_incidents):
    """Generates insights for each group."""
    insights = []
    for key, incidents in grouped_incidents.items():
        # Handle the case where tokens or category have commas or special characters
        # Split the key by a safe delimiter like "|" and avoid multiple delimiters
        key_parts = re.split(r"[,|-]+", key)
        
        if len(key_parts) < 2:
            raise ValueError(f"Unexpected key format: {key}")
        
        tokens = key_parts[0]
        category = key_parts[-1]
        
        # Generate insights for the group
        urgent_count = sum(1 for i in incidents if i.category.lower() == 'urgent')
        locations = {i.location for i in incidents}
        insights.append({
            "tokens": tokens,
            "category": category,
            "urgent_cases": urgent_count,
            "locations": list(locations),
            "insight": f"{len(locations)} unique locations with {urgent_count} urgent cases."
        })
    return insights

@app.route('/top_incidents', methods=['GET'])
def top_incidents():
    # Group incidents by location, category, and timestamp
    incidents = Incident.query.all()
    
    # Aggregation
    location_counts = {}
    category_counts = {}
    date_counts = {}

    for incident in incidents:
        location = incident.location
        category = incident.category
        date = incident.timestamp.date()
        
        location_counts[location] = location_counts.get(location, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        date_counts[date] = date_counts.get(date, 0) + 1
    
    # Sort by frequency
    sorted_location_counts = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_category_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_date_counts = sorted(date_counts.items(), key=lambda x: x[1], reverse=True)

    return render_template('top_incidents.html', 
                           location_counts=sorted_location_counts,
                           category_counts=sorted_category_counts,
                           date_counts=sorted_date_counts)




#GENERATE REPORTS FOR REPORT.HTML
from collections import defaultdict
from datetime import datetime, timedelta
from flask import request, render_template
from sqlalchemy.orm import joinedload
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datetime import datetime
# Download stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize sentence transformer model (supports multilingual, including Tagalog)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Define a list of Tagalog stopwords (can be extended further)
tagalog_stopwords = [
    'ang', 'ng', 'sa', 'at', 'ay', 'ako', 'ikaw', 'siya', 'kami', 'sila', 'ito', 'iyan', 'iyon', 'mga', 'naman', 
    'kasi', 'kung', 'saan', 'paano', 'kaya', 'bawat', 'lahat', 'wala', 'may', 'para', 'dahil', 'dapat', 
    'na'
]

# Combine English and Tagalog stopwords
stop_words = set(stopwords.words('english')).union(set(tagalog_stopwords))

def remove_stopwords(text):
    """Remove stopwords from the text."""
    text = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]  # Ignore short words
    return " ".join(filtered_words)

def encode_text(text):
    """Encode the text using sentence transformer to get embeddings."""
    return np.array(model.encode([text]))  # Ensure the output is a numpy array for cosine similarity

@app.route('/reports', methods=['GET'])
def reports():
    """
    Generate and render incident reports with advanced search and natural language processing.
    """
    # Get filter inputs
    incident_id = request.args.get('id')
    category = request.args.get('category')
    location = request.args.get('location')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    report_text = request.args.get('report_text')
    search_query = request.args.get('nl_search')  # Natural language search query
    incident_type = request.args.get('type')
    tag = request.args.get('tag')

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    # Base query
    query = Incident.query

    # Apply advanced filters
    if incident_id:
        query = query.filter(Incident.id == incident_id)
    if category:
        query = query.filter(Incident.category.ilike(f"%{category}%"))
    if location:
        query = query.filter(Incident.location.ilike(f"%{location}%"))
    if start_date:
        query = query.filter(Incident.timestamp >= start_date)
    if end_date:
        query = query.filter(Incident.timestamp <= end_date)
    if report_text:
        query = query.filter(Incident.report_text.ilike(f"%{report_text}%"))
    if incident_type:
        query = query.filter(Incident.type == incident_type)
    if tag:
        query = query.filter(Incident.tag == tag)

    # Exclude 'blotter' type records
    query = query.filter(Incident.type != 'blotter')

    # Apply natural language search
    if search_query:
        import spacy
        from nltk.corpus import stopwords
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Load NLP model
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(search_query)

        # Extract keywords and entities
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        entities = [ent.text for ent in doc.ents]

        # Combine keywords and entities, remove duplicates and stopwords
        search_terms = set(keywords + entities)
        stop_words = set(stopwords.words('english'))
        filtered_search_terms = [term for term in search_terms if term.lower() not in stop_words]

        # Query the database with the filtered search terms
        if filtered_search_terms:
            all_incidents = query.all()  # Get all incidents based on previous filters
            all_reports = [i.report_text for i in all_incidents]

            # Use TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_reports + [' '.join(filtered_search_terms)])

            # Compute cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

            # Sort incidents by similarity
            similar_indices = cosine_sim.argsort()[0][-5:][::-1]  # Top 5 most similar
            filtered_incidents = [all_incidents[i] for i in similar_indices]
        else:
            filtered_incidents = []
    else:
        filtered_incidents = query.order_by(Incident.timestamp.desc()).paginate(
            page=page, per_page=per_page
        ).items

    # Pagination
    if search_query:
        total_filtered = len(filtered_incidents)
        paginated_incidents = filtered_incidents[(page - 1) * per_page: page * per_page]
        pagination = {
            'total': total_filtered,
            'page': page,
            'per_page': per_page,
            'pages': (total_filtered // per_page) + (1 if total_filtered % per_page > 0 else 0),
        }
    else:
        paginated_query = query.paginate(page=page, per_page=per_page)
        paginated_incidents = paginated_query.items
        pagination = {
            'total': paginated_query.total,
            'page': paginated_query.page,
            'per_page': paginated_query.per_page,
            'pages': paginated_query.pages,
        }

    # Render the template with results
    return render_template(
        'reports.html',
        incidents=paginated_incidents,
        pagination=pagination,
        id=incident_id,
        category=category,
        location=location,
        start_date=start_date,
        end_date=end_date,
        report_text=report_text,
        nl_search=search_query,
        type=incident_type,
        tag=tag,
        per_page=per_page
    )

@app.route('/blotter_reports', methods=['GET'])
def blotter_reports():
    """
    Generate and render blotter reports.
    """
    # Pagination parameters
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=10, type=int)

    # Get blotter incidents from the database
    blotter_incidents = Incident.query.filter(Incident.type == 'blotter').order_by(Incident.timestamp.desc()).all()

    # Pagination logic
    pagination = None
    if blotter_incidents:
        pagination = {
            'total': len(blotter_incidents),
            'page': page,
            'per_page': per_page,
            'pages': (len(blotter_incidents) // per_page) + 1,
            'items': blotter_incidents[(page - 1) * per_page: page * per_page]
        }

    # Render the template with results
    return render_template(
        'blotter_reports.html',
        incidents=pagination['items'] if pagination else blotter_incidents,
        pagination=pagination
    )

@app.route('/search', methods=['GET'])
def search():
    try:
        search_query = request.args.get('search_query', '')
        
        # Default to empty list if no search query is provided
        incidents = []
        pagination = None  # Initialize pagination variable
        
        if search_query:
            # Preprocess the search query using spaCy for NER and tokenization
            doc = nlp(search_query)
            
            # Extract keywords (nouns and proper nouns) and named entities from the query
            keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            entities = [ent.text for ent in doc.ents]
            
            # Combine keywords and entities for search criteria (remove duplicates)
            search_terms = set(keywords + entities)
            
            # Remove stopwords from search terms
            stop_words = set(stopwords.words('english'))
            filtered_search_terms = [term for term in search_terms if term.lower() not in stop_words]

            # Query the database with the filtered search terms using TF-IDF and Cosine Similarity
            if filtered_search_terms:
                # Extract report texts from all incidents
                all_incidents = Incident.query.all()
                all_reports = [i.report_text for i in all_incidents]

                # Vectorize the report text and the search query
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_reports + [' '.join(filtered_search_terms)])
                
                # Compute cosine similarity between the query and the incident reports
                cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                
                # Sort incidents by similarity to the search query
                similar_indices = cosine_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar incidents
                incidents = [all_incidents[i] for i in similar_indices]

        # Add pagination for the search results
        page = request.args.get('page', 1, type=int)  # Get the page number from query string
        per_page = 10  # Define how many results per page
        pagination = Pagination(page=page, total=len(incidents), per_page=per_page, record_name='incidents')

        # Paginate the incidents
        paginated_incidents = incidents[(page - 1) * per_page: page * per_page]

        # Safeguard: Ensure pagination object is not None and has valid attributes
        start_page = 1
        end_page = pagination.pages if pagination else 1
        if pagination and pagination.page:
            start_page = max(pagination.page - 2, 1)
            end_page = min(pagination.page + 2, pagination.pages)

        # Render the results in the dashboard template, passing pagination data
        return render_template(
            'dashboard.html',
            incidents=paginated_incidents,
            pagination=pagination,
            start_page=start_page,
            end_page=end_page,
            search_query=search_query  # Pass the search query to the template
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to get similar report texts using cosine similarity
def get_similar_reports2(input_report_text, analysis_records):
    # Ensure the input report_text is valid
    if not input_report_text:
        raise ValueError("Input report text is empty or invalid")

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Combine the input report text and the report texts from the analysis records
    text_data = [input_report_text] + [record.report_text for record in analysis_records if record.report_text]

    if not text_data:
        raise ValueError("No valid report texts provided for analysis")
    
    # Fit and transform the text data
    text_matrix = vectorizer.fit_transform(text_data)

    # Compute similarity between the input report text and each analysis record
    similarity_scores = cosine_similarity(text_matrix[0:1], text_matrix[1:])
    
    return similarity_scores


from flask_caching import Cache

# Initialize cache globally
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
# Configure the cache (simple cache for demonstration)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

# Initialize the Cache with the app
cache = Cache(app)

@cache.cached(timeout=300, key_prefix='incident_analysis')
def get_incident_analysis():
    """Fetch all analysis records from the database with caching."""
    return IncidentAnalysis.query.all()

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer, util
import spacy



# Load the pre-trained SentenceTransformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the spacy model for NLP preprocessing
nlp = spacy.load('en_core_web_sm')

# Define additional Tagalog stopwords
TAGALOG_STOPWORDS = {
    "ang", "mga", "sa", "ng", "ako", "ikaw", "siya", "sila", "kami", "kayo", 
    "ito", "iyon", "ay", "at", "na", "pero", "kung", "dahil", "para", "po", 
    "ho", "dito", "doon", "iyan", "may", "wala", "meron", "paano", "ngayon", 
    "noon", "bakit", "lahat", "hindi", "oo", "o", "nga", "naman", "pa", "ba",
    "lang", "ganun", "ganito", "diba", "eh", "kasi", "talaga"
}

def preprocess_text(text, stop_words, disregard_words):
    """
    Preprocesses text by removing stopwords, disregard words, and non-alphabetical tokens.
    
    Args:
        text (str): The input text to preprocess.
        stop_words (set): A set of stopwords to filter out.
        disregard_words (set): A set of words to disregard during matching.
    
    Returns:
        str: The preprocessed text.
    """
    doc = nlp(text)
    filtered_tokens = [
        token.lemma_ for token in doc 
        if token.is_alpha and token.text.lower() not in stop_words 
        and token.text.lower() not in disregard_words and not token.ent_type_
    ]
    return ' '.join(filtered_tokens)

def extract_keywords(text, stop_words, disregard_words):
    """
    Extracts significant keywords from a text, excluding stopwords and disregard words.
    """
    doc = nlp(text)
    return {
        token.text.lower() for token in doc 
        if token.is_alpha 
        and token.text.lower() not in stop_words 
        and token.text.lower() not in disregard_words  # Exclude disregard words
        and not token.ent_type_
    }

def keyword_filter(text1, text2, stop_words, disregard_words):
    """
    Checks for keyword overlap between two texts to verify semantic match, ignoring disregard words.
    
    Args:
        text1 (str): First text input.
        text2 (str): Second text input.
        stop_words (set): Stopwords to filter out.
        disregard_words (set): Words to disregard in keyword matching.
    
    Returns:
        bool: True if there is significant overlap in key terms, otherwise False.
    """
    keywords1 = extract_keywords(text1, stop_words, disregard_words)
    keywords2 = extract_keywords(text2, stop_words, disregard_words)
    
    # Check for intersection of key terms
    common_keywords = keywords1.intersection(keywords2)
    return len(common_keywords) > 0

def get_similar_reports(report_text, analysis_records, incident, threshold=0.7):
    """
    Finds similar reports using a hybrid approach: Sentence Transformers and keyword-based matching.
    
    Args:
        report_text (str): The text of the current incident report.
        analysis_records (list): A list of incident analysis records.
        incident (dict): The incident object containing disregard_words.
        threshold (float): Similarity score threshold for matching.
    
    Returns:
        list: A list of similar records with action points and scores.
    """
    if analysis_records is None:
        print("Error: analysis_records is None!")
        return []

    if not analysis_records:
        print("Debug: No analysis records found (empty list).")
        return []

    # Merge default stopwords with Tagalog-specific stopwords
    stop_words = set(nlp.Defaults.stop_words).union(TAGALOG_STOPWORDS)
    
    # Get disregard words from the incident object
    disregard_words = set(incident.disregard_words) if hasattr(incident, 'disregard_words') else set()
    print(f"Debug: Disregard words: {disregard_words}")

    # Preprocess the report text and analysis texts
    processed_report_text = preprocess_text(report_text, stop_words, disregard_words)
    
    # For each analysis record, preprocess and compute similarity
    similar_reports = []
    for record in analysis_records:
        if 'report_text' not in record:
            print(f"Error: Missing 'report_text' in record {record}")
            continue

        processed_analysis_text = preprocess_text(record['report_text'], stop_words, disregard_words)

        # Encode the processed texts using sentence transformer model
        query_embedding = model.encode([processed_report_text], convert_to_tensor=True)
        analysis_embedding = model.encode([processed_analysis_text], convert_to_tensor=True)

        # Compute cosine similarity
        cosine_score = util.pytorch_cos_sim(query_embedding, analysis_embedding)[0][0].item()

        # Extract keywords for comparison
        keywords = extract_keywords(report_text, stop_words, disregard_words)
        record_keywords = extract_keywords(record['report_text'], stop_words, disregard_words)

        # Debug keyword overlap
        print(f"Debug: Keywords1 = {keywords}, Keywords2 = {record_keywords}")

        # Filter based on either cosine similarity or keyword match
        if cosine_score >= threshold or keyword_filter(report_text, record['report_text'], stop_words, disregard_words):
            similar_reports.append({
                "text": record['report_text'],
                "action_points": record.get('action_points', ''),
                "score": cosine_score,
                "keyword_overlap": list(keywords.intersection(record_keywords))  # Optional: for debugging keyword matches
            })

    # Sort reports by similarity score (descending order)
    similar_reports.sort(key=lambda x: x['score'], reverse=True)
    
    return similar_reports

def get_incident_analysis():
    return IncidentAnalysis.query.all()  # This will return ORM objects

@app.route('/get_survey_questions/<int:survey_id>', methods=['GET'])
def get_survey_questions(survey_id):
    survey = Survey.query.get(survey_id)
    if not survey:
        return jsonify({'error': 'Survey not found'}), 404

    questions = []
    for question in survey.questions:
        questions.append({
            'id': question.id,
            'text': question.text,
            'input_method': question.input_method
        })

    return jsonify({'questions': questions})

@app.route('/incident/<int:incident_id>', methods=['GET', 'POST'])
def incident_details(incident_id):
    try:
        # Query the incident by ID
        incident = Incident.query.get_or_404(incident_id)
        user = USERS.query.filter_by(user_id=incident.user_id).first()
        user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
        
        # Fetch all incident analysis records and convert them to dictionaries
        incident_analysis_records = get_incident_analysis()
        incident_analysis_dicts = [analysis.to_dict() for analysis in incident_analysis_records]

        # Find similar reports using the optimized function
        similar_reports = get_similar_reports(report_text=incident.report_text, 
                                            analysis_records=incident_analysis_dicts,
                                            incident=incident,
                                            threshold=0.85)  # Adjusted threshold for higher accuracy

# Collect unique action points, ensuring no duplicates
        action_points = set()

# Check for exact matches first
        exact_matches = [report for report in incident_analysis_dicts if report['report_text'] == incident.report_text]

# Add action points from exact matches
        for report in exact_matches:
            if 'action_points' in report and report['action_points']:
                points = report['action_points'].split('\n')  # Assuming action points are newline-separated
                action_points.update([point.strip() for point in points if point.strip()])

# Add action points from similar reports
        for report in similar_reports:
            if 'action_points' in report and report['action_points']:
                points = report['action_points'].split('\n')
                action_points.update([point.strip() for point in points if point.strip()])

# Convert the set back to a sorted list for rendering
        action_points = sorted(action_points)

        # SIMILAR INCIDENTS LIST
        # Fetch all incidents for comparison
        all_incidents = Incident.query.all()

        # Extract report texts from all incidents
        all_reports = [i.report_text for i in all_incidents]

        # Vectorize the report text using TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_reports)

        # Get the index of the current incident in the list
        current_index = all_incidents.index(incident)

        # Calculate cosine similarity between the current incident and all others
        cosine_sim = cosine_similarity(tfidf_matrix[current_index], tfidf_matrix)

        # Get the most similar incidents based on cosine similarity
        similar_indices = cosine_sim.argsort()[0][-6:-1]  # Get top 5 similar incidents (excluding itself)

        # Prepare the data to be passed to the template
        similar_incidents = [all_incidents[i] for i in similar_indices]

        # Prepare the data for displaying the current incident
        incident_data = {
            'id': incident.id,
            'report_text': incident.report_text,
            'timestamp': incident.timestamp,
            'category': incident.category,
            'latitude': incident.latitude,
            'longitude': incident.longitude,
            'tokens': incident.tokens,
            'openai_analysis': incident.openai_analysis,
            'user_id': incident.user_id,
            'location': incident.location,
            'language': incident.language,
            'media_path': incident.media_path if incident.media_path else None
        }

        # Fetch responses for this incident, ordered by timestamp descending
        responses = Response.query.filter_by(incident_id=incident_id).order_by(Response.timestamp.desc()).all()
       
        matches = None
        if incident.media_path:
            matches = analyze_media(incident.media_path)

        # Handle adding a new response
        if request.method == 'POST':
            response_text = request.form['response']
            new_response = Response(
                user_id=session['user_id'],  # Assuming you have a current_user for logged-in user
                response=response_text,
                incident_id=incident.id,
                timestamp=datetime.utcnow()
            )
            db.session.add(new_response)
            db.session.commit()
            return redirect(url_for('incident_details', incident_id=incident_id))

        return render_template('incident_details.html', incident=incident_data, 
                               similar_incidents=similar_incidents, responses=responses, 
                               action_points=action_points, matches=matches, user_name=user_name)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def analyze_media(media_path):
    folder_paths = ['./static/uploads', './static/photos/POI']  # Folders to search for matches
    matches = []

    # Helper function to calculate similarity
    def calculate_similarity(img1, img2):
        diff = cv2.absdiff(img1, img2)
        return np.mean(diff)  # Lower scores indicate higher similarity

    # Helper function for face detection and matching
    def detect_and_match_faces(target_img, compare_img, media_type='photo'):
        # Load the pre-trained face detector (Haar Cascade or a deep learning model)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert images to grayscale for face detection
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        compare_gray = cv2.cvtColor(compare_img, cv2.COLOR_BGR2GRAY)

        # Detect faces in both images
        target_faces = face_cascade.detectMultiScale(target_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        compare_faces = face_cascade.detectMultiScale(compare_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If faces are detected, extract the face region and compare
        if len(target_faces) > 0 and len(compare_faces) > 0:
            for (x, y, w, h) in target_faces:
                target_face = target_img[y:y+h, x:x+w]
                for (cx, cy, cw, ch) in compare_faces:
                    compare_face = compare_img[cy:cy+ch, cx:cx+cw]
                    
                    # Resize both faces for consistent comparison
                    target_face_resized = cv2.resize(target_face, (256, 256), interpolation=cv2.INTER_AREA)
                    compare_face_resized = cv2.resize(compare_face, (256, 256), interpolation=cv2.INTER_AREA)

                    # Calculate similarity score between faces
                    score = calculate_similarity(target_face_resized, compare_face_resized)

                    if score < 20:  # Threshold for similarity
                        return True
        return False

    # Helper function to search a folder for matches
    def search_folder(folder, target_img=None, media_type='photo'):
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if media_type == 'photo' and file_name.endswith(('.jpg', '.jpeg', '.png')):
                compare_img = cv2.imread(file_path)
                if compare_img is None:
                    print(f"Failed to read image: {file_path}")
                    continue
                
                if detect_and_match_faces(target_img, compare_img):
                    matches.append({
                        'type': 'photo',
                        'file_name': file_name,
                        'link': url_for('static', filename=os.path.relpath(file_path, './static').replace('\\', '/'))
                    })

            elif media_type == 'video' and file_name.endswith('.mp4'):
                compare_cap = cv2.VideoCapture(file_path)
                compare_frame_count = 0
                while True:
                    ret_compare, compare_frame = compare_cap.read()
                    if not ret_compare:
                        break
                    if compare_frame_count % 30 == 0:
                        compare_frame_gray = cv2.cvtColor(compare_frame, cv2.COLOR_BGR2GRAY)
                        if detect_and_match_faces(target_img, compare_frame_gray):
                            matches.append({
                                'type': 'video',
                                'file_name': file_name,
                                'link': url_for('static', filename=os.path.relpath(file_path, './static').replace('\\', '/'))
                            })
                    compare_frame_count += 1
                compare_cap.release()

    # Analyze photo
    if media_path.endswith(('.jpg', '.jpeg', '.png')):
        target_img_path = media_path
        print(f"Target image path: {target_img_path}")
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            print(f"Error: Could not read target image {target_img_path}")
            return matches

        for folder in folder_paths:
            if os.path.exists(folder):
                print(f"Searching in folder: {folder}")
                search_folder(folder, target_img=target_img, media_type='photo')
            else:
                print(f"Folder does not exist: {folder}")

    # Analyze video
    elif media_path.endswith('.mp4'):
        target_video_path = os.path.join('./static', media_path)
        print(f"Target video path: {target_video_path}")
        cap = cv2.VideoCapture(target_video_path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for folder in folder_paths:
                    if os.path.exists(folder):
                        search_folder(folder, target_img=frame_gray, media_type='video')
            frame_count += 1
        cap.release()

    print(f"Matches found: {matches}")
    return matches




from scipy.spatial.distance import euclidean
from mtcnn import MTCNN  # MTCNN for face detection
import dlib
import cv2
import os
import numpy as np
from flask import jsonify

# Initialize MTCNN detector and Dlib's face recognition model
detector = MTCNN()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Function to detect faces using MTCNN
def detect_faces_mtcnn(image):
    if image is None or len(image.shape) < 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a valid RGB image.")
    detections = detector.detect_faces(image)
    return [(d['box'][0], d['box'][1], d['box'][2], d['box'][3]) for d in detections]

# Function to get normalized face embeddings
def get_face_embedding(image, face_rect):
    rect = dlib.rectangle(face_rect[0], face_rect[1], face_rect[0] + face_rect[2], face_rect[1] + face_rect[3])
    shape = sp(image, rect)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor) / np.linalg.norm(face_descriptor)

# Function to calculate similarity using Euclidean distance
def calculate_face_similarity(embedding1, embedding2):
    return euclidean(embedding1, embedding2)

# Function to compare two images
def compare_images(img1, img2):
    faces1 = detect_faces_mtcnn(img1)
    faces2 = detect_faces_mtcnn(img2)

    if not faces1 or not faces2:
        return None  # No faces detected in either image

    embeddings1 = [get_face_embedding(img1, rect) for rect in faces1]
    embeddings2 = [get_face_embedding(img2, rect) for rect in faces2]

    similarities = [calculate_face_similarity(emb1, emb2) for emb1 in embeddings1 for emb2 in embeddings2]
    # Return the best (min) similarity score
    return min(similarities, default=None)

# Function to analyze media (photos or videos)
def analyze_media2(media, media_type='photo'):
    folder_paths = ['./static/uploads', './static/photos/POI']
    matches = []

    if media_type == 'photo':
        for folder_path in folder_paths:
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img_to_compare = cv2.imread(img_path)

                if img_to_compare is None or len(img_to_compare.shape) < 3 or img_to_compare.shape[2] != 3:
                    continue

                similarity_score = compare_images(media, img_to_compare)
                if similarity_score is not None:  # Include all matches, even similarity score 0.0
                    matches.append({'filename': filename, 'similarity': similarity_score})

    elif media_type == 'video':
        video_capture = cv2.VideoCapture(media)
        frame_count = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_count % 30 == 0:  # Process every 30th frame
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except:
                    continue

                for folder_path in folder_paths:
                    for filename in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, filename)
                        img_to_compare = cv2.imread(img_path)

                        if img_to_compare is None or len(img_to_compare.shape) < 3 or img_to_compare.shape[2] != 3:
                            continue

                        similarity_score = compare_images(frame, img_to_compare)
                        if similarity_score is not None:  # Include all matches
                            matches.append({'filename': filename, 'similarity': similarity_score})

            frame_count += 1
        video_capture.release()

    # Filter matches to only include similarity score <= 0.3 (or 0.0)
    print("Matches before filtering:", matches)
    matches = [match for match in matches if match['similarity'] <= 0.45]
    print("Matches after filtering:", matches)

    # Sort matches by similarity in ascending order (lower score = better .match)
    matches = sorted(matches, key=lambda x: x['similarity'])

    return matches
from io import BytesIO
import tempfile

from flask import render_template, request, flash

# Store progress globally (this could be session or a more sophisticated solution)
processing_progress = 0

@app.route('/upload_media', methods=['GET', 'POST'])
def upload_media():
    global processing_progress
    matches = []
    
    # Reset progress when a new request is received
    processing_progress = 0

    if request.method == 'POST':
        photo = request.files.get('photo')
        video = request.files.get('video')

        # Process photo if uploaded
        if photo and allowed_file(photo.filename):
            photo_data = photo.read()  # Read photo into memory
            
            if not photo_data:
                flash('Error: Empty photo file.', 'danger')
            else:
                photo_img = cv2.imdecode(np.frombuffer(photo_data, np.uint8), cv2.IMREAD_COLOR)
                
                if photo_img is None or len(photo_img.shape) != 3 or photo_img.shape[2] != 3:
                    flash('Error: Invalid photo format. Please upload a valid color image.', 'danger')
                else:
                    matches = analyze_media2(photo_img, media_type='photo')
                    print(f"Matches Data: {matches}")  # Debug print for matches

                    # Simulate processing progress
                    for i in range(10):
                        time.sleep(0.2)  # Simulate some processing time
                        processing_progress = i * 10  # Update progress

        # Process video if uploaded
        if video and allowed_file(video.filename):
            video_data = video.read()  # Read video into memory
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_data)
                tmp_video_path = tmp_video.name

            try:
                video_cap = cv2.VideoCapture(tmp_video_path)
                if not video_cap.isOpened():
                    flash('Error: Couldn\'t open the video.', 'danger')
                else:
                    matches = analyze_media2(video_cap, media_type='video')
                    print(f"Matches Data: {matches}")  # Debug print for matches

                    # Simulate processing progress
                    for i in range(10):
                        time.sleep(0.2)  # Simulate some processing time
                        processing_progress = i * 10  # Update progress
            finally:
                if os.path.exists(tmp_video_path):
                    os.remove(tmp_video_path)

        # Filter out matches with similarity 0.0 but still show them
        matches = [match for match in matches if match['similarity'] > 0.0 or match['similarity'] == 0.0]

        # Sort matches by similarity in ascending order (lower similarity = better match)
        matches = sorted(matches, key=lambda x: x['similarity'])

        # Construct `link` for each match
        folder_path = '/static/uploads/'  # Default folder for uploads
        POI_folder_path = os.path.join(app.root_path, 'static/photos/POI/')  # POI folder path on the server

        for match in matches:
            filename = match.get('filename')
            print(f"Checking for file: {filename}")  # Debug print for filenames
            
            if filename:
                full_POI_path = os.path.join(POI_folder_path, filename)

                # Check if the file exists in POI folder
                if os.path.exists(full_POI_path):
                    print(f"Found {filename} in POI folder.")
                    match['link'] = f"/static/photos/POI/{filename}"  # Use POI folder path for link
                else:
                    print(f"{filename} NOT found in POI folder.")
                    match['link'] = f"{folder_path}{filename}"  # Use regular uploads folder for missing files

    return render_template('upload_media.html', matches=matches)


@app.route('/get_progress')
def get_progress():
    return jsonify(progress=processing_progress)


@app.route('/search_incidents')
def search_incidents():
    # Get the media_path (or file name) from the request
    media_path = request.args.get('media_path')
    
    if media_path.startswith('/'):
        media_path = media_path[8:]
        print(media_path)

    if not media_path:
        return render_template('no_match.html', message="No media path provided.")
    
    # Search the Incident table for matching media_path
    incident = Incident.query.filter(Incident.media_path.ilike(f"%{media_path}%")).first()
    
    if incident:
        return render_template('incident_details.html', incident=incident)
    else:
        return render_template('no_match.html', message="No incidents found matching the media.")
    

@app.route('/search_person_of_interest')
def search_person_of_interest():
    # Get the photo_path from the query parameter
    photo_path = request.args.get('photo_path')

    # Remove the first slash from the photo_path, if it exists
    if photo_path.startswith('/'):
        photo_path = photo_path[1:]
        photo_slashed = photo_path[8:]

    if photo_path:
        # Query the PersonOfInterest table based on the photo_path
        person = PersonOfInterest.query.filter_by(photo_path=photo_path).first()

        if person:
            # If a match is found, render a template with person details
            return render_template('person_details.html', person=person, photo_path=photo_slashed)
        else:
            # If no match is found, show a message
            return render_template('no_match.html', photo_path=photo_path, person=person)
    else:
        return "No photo path provided", 400
    
# API route to submit incident
@app.route('/report', methods=['POST'])
def report():
    try:
        # Extract form data
        report_text = request.form.get('report_text')
        latitude = request.form.get('latitude')
        user_id = request.form.get('user_id')
        longitude = request.form.get('longitude')

        print("Form Data:", request.form)
        print("Files Data:", request.files)

        if not report_text or not latitude or not longitude:
            return jsonify({"error": "Report text, latitude, and longitude are required"}), 400

        latitude = float(latitude)
        longitude = float(longitude)

        # Handle optional media upload
        import uuid
        from werkzeug.utils import secure_filename

        # Handle optional media upload
        media_path = None
        if 'media' in request.files and request.files['media'].filename != '':
            media_file = request.files['media']
            if allowed_file(media_file.filename):
                # Secure the original filename
                original_filename = secure_filename(media_file.filename)
                base, ext = os.path.splitext(original_filename)  # Separate the name and extension

                # Generate a unique filename if the file already exists
                unique_filename = original_filename
                upload_folder = app.config['UPLOAD_FOLDER']
                os.makedirs(upload_folder, exist_ok=True)  # Ensure the upload folder exists

                while os.path.exists(os.path.join(upload_folder, unique_filename)):
                    unique_filename = f"{base}_{uuid.uuid4().hex}{ext}"  # Append a unique identifier

                # Save the file to the unique path
                media_path = os.path.join(upload_folder, unique_filename)
                media_file.save(media_path)
            else:
                return jsonify({"error": "Invalid file type"}), 400

        # Categorize the report
        category = categorize_incident(report_text)
        print("Incident Category:", category)  # Moved after category is assigned
        tokens = ", ".join([w for w in word_tokenize(report_text) if w.lower() not in stopwords.words('english')])
        openai_analysis = "Analysis pending."
        #openai_analysis = analyze_report(report_text)
        location = get_location(latitude, longitude)
        language = detect(report_text)

        # Create and commit incident
        new_incident = Incident(
            report_text=report_text,
            media_path=media_path,
            latitude=latitude,
            longitude=longitude,
            category=category,
            tokens=tokens,
            openai_analysis=openai_analysis,
            location=location,
            user_id = user_id,
            language=language,
            type="citizen-online"
        )

        print("New Incident:", vars(new_incident))  # Debug: Check the incident object
        
        db.session.add(new_incident)
        db.session.commit()

        print("Incident saved with ID:", new_incident.id)  # Check if ID was generated

        return jsonify({"message": "Report submitted", "category": category}), 201

    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude"}), 400
    except Exception as e:
        db.session.rollback()
        print(f"Database error: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

# Dashboard to view analysis

from math import ceil

from sqlalchemy import func



@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))

        if session['role'] != 'ADMIN':
            flash('You do not have the required permissions to access this page.', 'danger')
            return redirect(url_for('login'))

        # Get the start and end date from the query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        page = request.args.get('page', 1, type=int)  # Get the page number, default is 1
        per_page = request.args.get('per_page', 10, type=int)  # Get records per page, default is 10
        
        # Initialize filters
        start_date = None
        end_date = None
        
        # Convert the string date values to datetime objects if they are provided
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M')
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%dT%H:%M')

        # Query incidents from the database with date filtering if dates are provided
        query = Incident.query
        
        if start_date and end_date:
            query = query.filter(Incident.timestamp >= start_date, Incident.timestamp <= end_date)
        elif start_date:
            query = query.filter(Incident.timestamp >= start_date)
        elif end_date:
            query = query.filter(Incident.timestamp <= end_date)

        # Exclude 'blotter' type records
        query = query.filter(Incident.type != 'blotter')

        # Apply pagination: .paginate(page, per_page, error_out=False) returns a Pagination object
        pagination = query.order_by(Incident.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
        
        incidents = pagination.items
        total_incidents = pagination.total  # Get the total number of incidents for pagination
        
        if not incidents:
            return jsonify({"message": "No data available"}), 404

        # Prepare the data to be displayed in the dashboard using the to_dict() method
        dashboard_data = [incident.to_dict() for incident in incidents]

        # Get the predictive analysis data (this could be a call to a function or API)
        recent_reports = Incident.query.order_by(Incident.timestamp.desc()).limit(50).all()
        prediction = "Increased risk of vandalism in urban areas at night."  # Example prediction

        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)

        # Query for incidents today vs yesterday
        incidents_today = Incident.query.filter(func.date(Incident.timestamp) == today).count()
        incidents_yesterday = Incident.query.filter(func.date(Incident.timestamp) == yesterday).count()

        # Query for responses today
        responses_today = Response.query.filter(func.date(Response.timestamp) == today).count()

        # Query for the top category today
        top_category_today = db.session.query(Incident.category, func.count(Incident.id).label('count'))\
            .filter(func.date(Incident.timestamp) == today)\
            .group_by(Incident.category)\
            .order_by(func.count(Incident.id).desc())\
            .first()

        if top_category_today:
            top_category_today = top_category_today.category
        else:
            top_category_today = 'None'

        # Calculate the start and end pages to limit the number of page links displayed
        start_page = max(pagination.page - 2, 1)
        end_page = min(pagination.page + 2, pagination.pages)

        # Media path (if you need it for any specific use)
        media_path = Incident.media_path

        # Pass the data to the template, including pagination data
        return render_template(
            'dashboard.html',
            username=session['username'],
            incidents=dashboard_data,
            prediction=prediction,
            media_path=media_path,
            incidents_today=incidents_today,
            incidents_yesterday=incidents_yesterday,
            responses_today=responses_today,
            top_category_today=top_category_today,
            pagination=pagination,  # Ensure pagination is passed to the template
            total_incidents=total_incidents,
            per_page=per_page,
            start_page=start_page,
            end_page=end_page
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


    
# Generate sample data
@app.route('/generate_sample_data', methods=['POST'])
def generate_sample_data():
    sample_texts = [
        "Robbery occurred in the downtown area.",
        "A car was vandalized last night.",
        "Victim reported harassment in the workplace.",
        "Suspicious activity reported near the mall.",
        "A fraudulent check was cashed at the local bank.",
        "An assault was reported near the subway station."
    ]
    for _ in range(100):
        sample_text = random.choice(sample_texts)
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
        timestamp = datetime.utcnow() - timedelta(days=random.randint(0, 365))

        incident = Incident(
            report_text=sample_text,
            media_path=None,
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,  # Ensure this is a datetime object
            category=categorize_incident(sample_text),
            tokens=" ".join([w for w in word_tokenize(sample_text) if w.lower() not in stopwords.words('english')]),
            openai_analysis=analyze_report(sample_text),
            location=f"Latitude: {latitude}, Longitude: {longitude}",
        )
        db.session.add(incident)
    db.session.commit()
    return jsonify({"message": "100 sample incidents generated"}), 201

if __name__ == "__main__":
    app.run(debug=True)