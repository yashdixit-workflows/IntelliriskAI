import os
from google import genai

def get_gemini_client() -> genai.Client:
    """
    Initializes and returns a Google GenAI client.
    Uses multi-layer fallback:
    1. Checks if GEMINI_API_KEY environment variable is set.
    2. Fallback to GCP Vertex AI on the validated project 'tribal-pride-498113-d8'.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Warning: Failed to init client with GEMINI_API_KEY: {e}")
            
    # Fallback to Vertex AI project that is validated and active
    try:
        project_id = os.environ.get("GCP_PROJECT", "tribal-pride-498113-d8")
        return genai.Client(vertexai=True, project=project_id, location="us-central1")
    except Exception as e:
        print(f"Warning: Failed to init client with Vertex AI: {e}")
        # Default fallback
        return genai.Client()
