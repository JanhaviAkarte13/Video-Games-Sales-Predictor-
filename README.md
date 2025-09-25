Video Game Sales Predictor 🎮
A data science project that estimates video game sales based on a historical dataset. This project demonstrates a complete machine learning pipeline, from data preparation to a deployed web application.

🚀 Key Features
Sales Prediction: Use three machine learning models (Linear Regression and Random Forest) to predict a game's global sales.

Interactive Frontend: A user-friendly web interface built with Flask allows you to input a game's details and get instant predictions.

Data Visualization: Explore key trends in the historical video game market through interactive charts and dashboards.

Historical Analysis: Discover which genres, platforms, and publishers were the most successful over the years.

💡 Project Overview
This project addresses a fundamental business need in the video game industry: estimating a product's market potential. We built a predictive tool that uses a historical dataset to forecast a game's sales based on its characteristics, such as its name, genre, and platform.

The system is powered by a Flask backend that manages the data processing and model predictions. The results are then displayed on a simple, custom-built frontend, making the powerful predictive tool accessible to anyone.

🤖 Machine Learning Models
We employed three distinct models to provide a robust and reliable sales forecast:

Linear Regression: A foundational model used to establish a baseline prediction.

Random Forest: A more advanced model that combines the predictions of multiple decision trees to improve accuracy.

Using multiple models allows us to compare their results and have more confidence in our final sales estimate.

📈 Data & Visualizations
The dataset used for this project is from Kaggle. You can access it here: 

InsertKaggleDatasetLinkHere
.

Our visualizations provide valuable insights into the historical data:

Global Sales Over the Years: A line chart showing the total physical game sales, highlighting the peak of the market before the shift to digital distribution.

Top Selling Genres: A bar chart revealing the most commercially successful genres during the dataset's timeframe.

Platform Popularity: A visual representation of which gaming platforms had the most games released on them.

Popular Games Over the Years: A table listing the top-selling games from a specific era, demonstrating what was popular in the past.

Note: The project's predictions and visualizations are based on historical data and do not reflect current market trends, which are dominated by digital sales and subscription services.

🛠️ Setup & Installation
To run this project locally, follow these steps:

Clone the repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

Create a virtual environment:

python -m venv venv

Activate the virtual environment:

On Windows: venv\Scripts\activate

On macOS and Linux: source venv/bin/activate

Install the dependencies:

pip install Flask pandas scikit-learn joblib numpy matplotlib seaborn

Run the Flask application:

python app.py

The application will now be running on http://127.0.0.1:5000.
