from flask import Flask, jsonify, request
import time
from datetime import datetime

app = Flask(__name__)


# Simple model for demonstration
def model(x):
    return [1 if i > 10 else 0 for i in x]  # Handles lists of inputs


@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Flask app! Use the /predict endpoint for POST requests."


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for making predictions.
    Expects a JSON payload with an 'instances' key containing a list of values.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    try:
        # Parse request JSON
        request_json = request.get_json()
        if not request_json or "instances" not in request_json:
            return jsonify({"error": "Invalid input, 'instances' key required"}), 400

        # Extract input instances
        request_instances = request_json["instances"]

        if not isinstance(request_instances, list):
            return jsonify({"error": "'instances' must be a list"}), 400

        # Experimental: Start timing prediction
        prediction_start_time = time.time()
        current_timestamp = datetime.now().isoformat()

        # Run the model
        prediction = model(request_instances)

        # Experimental: End timing prediction
        prediction_end_time = time.time()
        prediction_latency = prediction_end_time - prediction_start_time

        # Create output
        output = {
            "predictions": [{"cluster": pred} for pred in prediction],
            "metadata": {
                "prediction_latency": prediction_latency,
                "timestamp": current_timestamp,
            },
        }
        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    ## update test
