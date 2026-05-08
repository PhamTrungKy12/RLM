from google import genai
client = genai.Client()
def call_llm(prompt: str) -> str:

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.

    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    )
    return response.text