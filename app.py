import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import numpy as np
import os
import secrets
from collections import defaultdict

# Generate a strong, random key
secret_key = secrets.token_hex(16)

# Load the dataset and models
df = None
try:
    if os.path.exists('vgsales.csv'):
        df = pd.read_csv('vgsales.csv')
        df.dropna(subset=['Year', 'Publisher', 'Genre', 'Platform'], inplace=True)
        df['Year'] = df['Year'].astype(int)
        df['Publisher'] = df['Publisher'].str.strip()
    else:
        print("Error: vgsales.csv not found.")

    os.makedirs('models', exist_ok=True)
except Exception as e:
    print(f"An error occurred while loading vgsales.csv: {e}")
    df = None

# Load the trained models and label encoders
def load_models():
    models = {}
    encoders = {}
    try:
        models['linear_regression'] = joblib.load('models/linear_regression_model.pkl')
        models['random_forest'] = joblib.load('models/random_forest_model.pkl')
        encoders['le_platform'] = joblib.load('models/le_platform.pkl')
        encoders['le_genre'] = joblib.load('models/le_genre.pkl')
        encoders['le_publisher'] = joblib.load('models/le_publisher.pkl')
        return models, encoders
    except FileNotFoundError:
        return None, None

app = Flask(__name__, static_url_path='/static')
app.secret_key = secret_key

# --- Authentication and Routes ---

users = {
    "user": "password",
    "admin": "admin123"
}

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if users.get(username) == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if df is None:
        return render_template('predict.html', error="Dataset not loaded. Please ensure vgsales.csv is in the project directory.")

    game_names = df['Name'].unique().tolist()
    
    if request.method == 'POST':
        game_name = request.form.get('game_name')
        if not game_name:
            return render_template('predict.html', error="Please select a game name.", game_names=game_names)

        game_data = df[df['Name'] == game_name].iloc[0]
        
        # Extract features for prediction
        platform = game_data['Platform']
        genre = game_data['Genre']
        publisher = game_data['Publisher']
        year = game_data['Year']
        actual_sales = game_data['Global_Sales']

        models, encoders = load_models()
        if not models or not encoders:
            return render_template('predict.html', error="Prediction models not loaded. Please run 'python train_models.py' first.", game_names=game_names)

        try:
            platform_encoded = encoders['le_platform'].transform([platform])[0]
            genre_encoded = encoders['le_genre'].transform([genre])[0]
            publisher_encoded = encoders['le_publisher'].transform([publisher])[0]
        except ValueError as e:
            return render_template('predict.html', error=f"Prediction failed: One of the selected values is not in the training data. Details: {e}", game_names=game_names)

        features = np.array([[platform_encoded, genre_encoded, publisher_encoded, year]])

        linear_pred = models['linear_regression'].predict(features)[0]
        rf_pred = models['random_forest'].predict(features)[0]

        predictions = {
            'linear_regression': round(linear_pred, 2),
            'random_forest': round(rf_pred, 2)
        }

        return render_template(
            'results.html',
            game_name=game_name,
            predictions=predictions,
            actual_sales=round(actual_sales, 2),
            platform=platform,
            genre=genre,
            publisher=publisher,
            year=year
        )
    
    return render_template('predict.html', game_names=game_names)

@app.route('/trending')
def trending():
    if 'username' in session:
        if df is None:
            return render_template('trending.html', error="Dataset not loaded. Please ensure vgsales.csv is in the project directory.")
        
        top_games = df.sort_values(by='Global_Sales', ascending=False).head(100)
        
        return render_template('trending.html', games=top_games.to_dict('records'))
    return redirect(url_for('login'))

@app.route('/visualize')
def visualize():
    if 'username' not in session:
        return redirect(url_for('login'))

    if df is None:
        return "Dataset not loaded. Please ensure vgsales.csv is in the project directory."
    
    sales_by_year = df.groupby('Year')['Global_Sales'].sum().reset_index()
    sales_data = {
        'labels': sales_by_year['Year'].tolist(),
        'datasets': [{
            'label': 'Global Sales (Millions)',
            'data': sales_by_year['Global_Sales'].tolist(),
            'borderColor': '#8b5cf6',
            'backgroundColor': 'rgba(139, 92, 246, 0.2)',
            'fill': True,
            'tension': 0.4
        }]
    }

    cleaned_df = df.copy()
    for col in ['Name', 'Publisher', 'Genre', 'Platform']:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].str.replace("'", "").str.replace('"', '')

    genres = cleaned_df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).head(5)
    genre_data = {
        'labels': genres.index.tolist(),
        'datasets': [{
            'label': 'Global Sales (Millions)',
            'data': genres.values.tolist(),
            'backgroundColor': ['#8b5cf6', '#a78bfa', '#c4b5fd', '#a5f3fc', '#2dd4bf'],
            'borderColor': 'rgba(255, 255, 255, 0.1)',
            'borderWidth': 1
        }]
    }

    platforms = cleaned_df['Platform'].value_counts().head(5)
    platform_data = {
        'labels': platforms.index.tolist(),
        'datasets': [{
            'label': 'Number of Games',
            'data': platforms.values.tolist(),
            'backgroundColor': ['#8b5cf6', '#a78bfa', '#c4b5fd', '#a5f3fc', '#2dd4bf'],
            'borderColor': 'rgba(255, 255, 255, 0.1)',
            'borderWidth': 1
        }]
    }

    top_games_by_year = cleaned_df.loc[cleaned_df.groupby('Year')['Global_Sales'].idxmax()][['Year', 'Name', 'Global_Sales']]
    top_games_data = top_games_by_year.to_dict('records')

    return render_template(
        'visualize.html', 
        sales_data=sales_data,
        genre_data=genre_data,
        platform_data=platform_data,
        top_games_data=top_games_data
    )

if __name__ == '__main__':
    if df is not None:
        print("Training models...")
        from train_models import train_and_save_models
        train_and_save_models()
    app.run(debug=True)


