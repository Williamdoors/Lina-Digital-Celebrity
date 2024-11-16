from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize NLP model
model = pipeline("text-generation", model="gpt-2")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("question")
    response = model(user_input, max_length=50, num_return_sequences=1)
    return jsonify({"response": response[0]["generated_text"]})

if __name__ == "__main__":
    app.run(debug=True)
