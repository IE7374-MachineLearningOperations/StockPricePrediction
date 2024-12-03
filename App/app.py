from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np

app = Flask(__name__)

# Placeholder for the prediction function (currently using random prediction for demo)
def predict_stock_price():
    return np.random.uniform(170, 180)  # Replace with actual prediction logic

# Generate stock graph for Google (GOOGL) using Plotly
def generate_plotly_graph():
    ticker = yf.Ticker('GOOGL')

    ## Fetch historical market data
    ## period = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    stock_data = ticker.history(period="10y")
    stock_data.reset_index(inplace=True)
    stock_data["Date"] = stock_data["Date"].dt.date
    stock_data.columns = stock_data.columns.str.lower()
    stock_data.columns = stock_data.columns.str.replace(" ", "_")

    # Create a line chart with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock_data['date'],  # Dates on the x-axis
        y=stock_data['close'],  # Closing prices on the y-axis
        mode='lines',
        name='GOOGL Close Price',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='GOOGL Stock Price (Past 10 Years)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )

    # Render the graph as an HTML div string
    graph_html = pio.to_html(fig, full_html=False)

    return graph_html


# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Visualization page
@app.route('/visualize')
def visualize():
    stock_graph = generate_plotly_graph()  # Generate the stock graph
    return render_template('visualize.html', stock_graph=stock_graph)

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    predicted_price = predict_stock_price()  # Placeholder prediction
    return render_template('predict.html', predicted_price=predicted_price)

# Reset and go back to home page
@app.route('/reset')
def reset():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)