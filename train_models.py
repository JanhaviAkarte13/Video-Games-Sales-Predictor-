import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import warnings

# Suppress all warnings for a cleaner output
warnings.filterwarnings('ignore')

def train_and_save_models():
    """
    Loads the video game sales data, preprocesses it, and trains
    Linear Regression and Random Forest models. The trained models are
    then saved to the 'models' directory.
    """
    try:
        # Load the dataset
        df = pd.read_csv('vgsales.csv')
        
        # Data cleaning and preprocessing
        # Drop rows with missing values
        df.dropna(inplace=True)
        # Drop rows with unknown years
        df = df[df['Year'] != 'N/A']
        # Convert year to integer
        df['Year'] = df['Year'].astype(int)
        
        # Feature engineering
        
        # Select features and target
        features = ['Platform', 'Genre', 'Publisher', 'Year']
        target = 'Global_Sales'
        
        # Preprocessing: Encoding categorical variables
        le_platform = LabelEncoder()
        le_genre = LabelEncoder()
        le_publisher = LabelEncoder()
        
        df['Platform_Encoded'] = le_platform.fit_transform(df['Platform'])
        df['Genre_Encoded'] = le_genre.fit_transform(df['Genre'])
        df['Publisher_Encoded'] = le_publisher.fit_transform(df['Publisher'])
        
        X = df[['Platform_Encoded', 'Genre_Encoded', 'Publisher_Encoded', 'Year']]
        y = df[target]
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- Train Models ---
        print("Training models...")
        
        # Linear Regression
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, y_train)
        print("Linear Regression model trained.")
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        print("Random Forest Regressor model trained.")
        
        # --- Save Models and Encoders ---
        print("Saving models and encoders...")
        joblib.dump(lin_reg_model, 'models/linear_regression_model.pkl')
        joblib.dump(rf_model, 'models/random_forest_model.pkl')
        joblib.dump(le_platform, 'models/le_platform.pkl')
        joblib.dump(le_genre, 'models/le_genre.pkl')
        joblib.dump(le_publisher, 'models/le_publisher.pkl')
        
        print("Models and encoders saved successfully.")
        
    except FileNotFoundError:
        print("Error: The 'vgsales.csv' file was not found. Please make sure the file is in the project's root directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    train_and_save_models()
