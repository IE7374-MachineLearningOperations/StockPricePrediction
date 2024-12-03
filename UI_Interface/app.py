import streamlit as st
import yfinance as yf

# Define companies and their corresponding stock codes
companies = {
    "Google LLC (GOOGL)": "GOOGL"
}

# Function to fetch live stock price
def fetch_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="2d")  # Ensure enough data points to access last and second-last close prices
        
        # Check if data is empty
        if data.empty:
            st.error("No data available for this ticker. Please check the ticker symbol or try again later.")
            return None, None

        # Check if there are enough data points
        if len(data) < 2:
            st.error("Not enough data to determine the current and previous closing prices.")
            return None, None

        # Extract current and previous close prices
        current_price = data["Close"].iloc[-1]
        previous_close = data["Close"].iloc[-2]
        return current_price, current_price >= previous_close

    except Exception as e:
        st.error(f"Error fetching stock price: {e}")
        return None, None

# App title
st.title("US Stock Market")

# Sidebar for navigation
selected = st.sidebar.radio("Menue", ["HOME", "STOCKS"])

if selected == "HOME":
    st.subheader("Welcome to the Stock Selector App!")
    st.write("Use this app to select a company and fetch its stock ticker.")
    st.write("---")

elif selected == "STOCKS":
    st.subheader("Stock Price Prediction")

    # Dropdown to select a company
    selected_company = st.selectbox("Select a company:", list(companies.keys()))

    # Retrieve the stock ticker for the selected company
    selected_stocks = companies[selected_company]

    # Display the selected company name
    st.write(f"**Selected Company:** {selected_company}")

    # Fetch and display the current stock price
    st.write("---")
    st.write("**Stock Information:**")

    # Columns for Current Stock Price and Predicted Stock Price
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Current Stock Price")
        current_price, is_high = fetch_stock_price(selected_stocks)
        if current_price is not None:
            color = "green" if is_high else "red"
            st.markdown(f"<h3 style='color:{color};'>${current_price:.2f}</h3>", unsafe_allow_html=True)

    with col2:
        st.write("### Predicted Stock Price")
        predicted_price_placeholder = st.empty()
        predicted_price_placeholder.markdown("<h3 style='color:gray;'>--</h3>", unsafe_allow_html=True)

    # Predict button
    st.write("---")
    if st.button("Predict"):
        # Placeholder logic for prediction (update as needed)
        predicted_price = 150.00  # Replace with actual prediction logic
        predicted_price_placeholder.markdown(f"<h3 style='color:blue;'>${predicted_price:.2f}</h3>", unsafe_allow_html=True)
