import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System message to guide AI behavior
SYSTEM_MESSAGE = """You are a helpful AI assistant that provides clear, accurate, 
and concise responses while maintaining a friendly tone."""


def get_openai_response(prompt):
    """
    Get a response from OpenAI's API
    """
    try:
        # Create chat completion with the new client
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        # Extract the response content
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# Create a simple Gradio interface
demo = gr.Interface(
    fn=get_openai_response,
    inputs=gr.Textbox(
        label="Ask me anything", placeholder="Type your question here...", lines=2
    ),
    outputs=gr.Textbox(label="AI Response", lines=4),
    title="💬 Simple ChatGPT Interface",
    description="Ask a question and get an answer from OpenAI's GPT-3.5-turbo model",
    examples=[
        ["What is Python programming?"],
        ["Tell me a short joke"],
        ["Write a haiku about programming"],
    ],
    theme=gr.themes.Soft(),  # Added a modern theme
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
