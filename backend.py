import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ai_agent import get_response_from_ai_agent
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.before_request
def log_request_info():
    logger.debug(f"Headers: {request.headers}")
    logger.debug(f"Body: {request.get_data()}")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "GroqSage AI"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        logger.info("Received chat request")
        data = request.get_json()
        
        if not data:
            logger.error("No data received in request")
            return jsonify({"error": "No data provided"}), 400
            
        if not isinstance(data.get("messages"), list) or len(data["messages"]) == 0:
            logger.error("Invalid messages format")
            return jsonify({"error": "Messages must be a non-empty array"}), 400

        logger.debug(f"Processing request with data: {data}")
        
        response = get_response_from_ai_agent(
            llm_id=data.get("model_name", "llama3-70b-8192"),
            query=data["messages"][0],
            allow_search=data.get("allow_search", False),
            system_prompt=data.get("system_prompt", "You are a helpful AI assistant."),
            provider=data.get("model_provider", "Groq")
        )
        
        logger.info("Successfully generated AI response")
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to process request",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    # Validate environment
    if not os.getenv("GROQ_API_KEY"):
        logger.critical("Missing GROQ_API_KEY in environment variables")
        exit(1)

    # Test Groq connection
    try:
        from langchain_groq import ChatGroq
        test = ChatGroq(model_name="llama3-70b-8192").invoke("test")
        logger.info("✅ Groq connection test successful")
    except Exception as e:
        logger.critical(f"❌ Groq connection failed: {e}")
        exit(1)
        
    app.run(host="0.0.0.0", port=9999, debug=True)