from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
from google.cloud import aiplatform
import numpy as np
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../StockPricePrediction/GCP/application_default_credentials.json"

app = Flask(__name__)

# Google Cloud Vertex AI settings
PROJECT_ID = 'striped-graph-440017-d7'
REGION = 'us-east1'  # e.g., 'us-central1'
ENDPOINT_ID = '7737663546892222464'

# Initialize Vertex AI client
def init_vertex_ai():
    aiplatform.init(project=PROJECT_ID, location=REGION)

# Function to call Vertex AI endpoint for stock price prediction
def predict_stock_price_from_vertex():
    init_vertex_ai()

    # Example instance to pass into the model
    instance = [2.776448861,2.75447283,2.812352314,-0.794546461,0,-0.020756897,0.270228793,-0.146470934,0.175553248,0.790922987,0.005361617,2.462872406,0.012411167,3.395914313,-1.357731101,0.249635018,3.25006235,1.684754563,1.940087983,1.929207385,0.450072396,0.579226308,2.733704725,-0.704395396,-0.230658202,-0.541495821,-1.370568575,0.915968439,-0.138979095,-0.137360564,-0.135724179,-0.030970336,2.776562648,2.708252797,2.716533926,2.701646201,2.697424831,2.676352776,2.739899379,2.669352908,2.683592403,2.749313023,2.712444883,2.727955169,2.737728958,2.730522236,0.229011846,1.227923433,4.120540266,1.257303426,1.834765223,2.582013389,1.711810826,2.430058907,-1.01315732,-0.25059627,0.223515342,2.861766746,2.854575289,2.85854613]

    # Get the endpoint object from Vertex AI
    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

    # Make the prediction request
    response = endpoint.predict(instances=[instance])

    # Return the predicted value
    return round(response.predictions[0],2)

# Function to get current stock price for Google (GOOGL)
def get_current_stock_price():
    ticker = yf.Ticker('GOOGL')
    todays_data = ticker.history(period='1d')
    return round(todays_data['Close'][0], 2)

def generate_plotly_graph():
    ticker = yf.Ticker('GOOGL')

    # Fetch historical market data
    stock_data = ticker.history(period="10y")
    stock_data.reset_index(inplace=True)
    stock_data["Date"] = stock_data["Date"].dt.date

    # Create a line chart with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='GOOGL Close Price'))

    # Update layout with precise hover format
    fig.update_layout(
        title='GOOGLE Stock Price (Past 10 Years)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        xaxis=dict(
            tickformat='%Y',  # Format date as "Nov 17, 2024"
            hoverformat='%b %d, %Y'
        )
    )

    # Render the graph as an HTML div string
    return pio.to_html(fig, full_html=False)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Visualization page
@app.route('/visualize')
def visualize():
    stock_graph = generate_plotly_graph()
    return render_template('visualize.html', stock_graph=stock_graph)

# Prediction page
@app.route('/predict')
def predict():
    current_price = get_current_stock_price()
    predicted_price = predict_stock_price_from_vertex()
    
    # Pass both the current and predicted prices to the template
    return render_template('predict.html', current_price=current_price, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)

