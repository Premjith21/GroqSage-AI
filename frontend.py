import streamlit as st
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App config
st.set_page_config(
    page_title="GroqSage AI",
    layout="centered",
    page_icon="🤖"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI Elements
st.title("GroqSage AI 🚀")
st.caption("Create and Interact with your Smart AI Agent")

with st.form("chat_form"):
    system_prompt = st.text_area(
        "Define your AI Agent:",
        value="You are an AI trained in computer science and programming. Provide concise, accurate answers with code examples when relevant.",
        height=100
    )
    
    user_query = st.text_area(
        "Enter your query:",
        placeholder="Explain how Python's async/await works with a real-world analogy and code snippet.",
        height=150
    )
    
    submitted = st.form_submit_button("Ask Agent!")

# On form submission
if submitted and user_query.strip():
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Generating response..."):
        try:
            payload = {
                "system_prompt": system_prompt,
                "messages": [user_query],
                "model_name": "llama3-70b-8192",
                "model_provider": "Groq",
                "allow_search": False  # Set to True if you have Tavily API key
            }
            
            logger.info(f"Sending request with payload: {payload}")
            response = requests.post(
                "https://groqsage-ai.onrender.com/chat",
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": data["response"]
                })
            else:
                st.error(f"Unexpected response format: {data}")
                logger.error(f"Unexpected response: {data}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
            logger.error(f"API request failed: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}")

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])