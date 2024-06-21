from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.models import Model
from keras.layers import Input, Dense
import tensorflow as tf
import datetime
import pickle
from model import recommend_for_user

app = Flask(__name__, static_url_path='/static')

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define API routes
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        print("Received data:", data)  # Log received data

        if 'user_id' not in data:
            return jsonify({'error': 'user_id is missing'}), 400
        
        user_id = int(data['user_id'])
        print("Parsed user_id:", user_id)  # Log parsed user_id

        recommendations = recommend_for_user(user_id)
        if recommendations.empty:
            return jsonify({'recommendations': []}), 200

        recommendations_json = recommendations.to_json(orient='records')
        return jsonify({'recommendations': recommendations_json}), 200
    except Exception as e:
        print(f"Error: {e}")  # Print the error message
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
