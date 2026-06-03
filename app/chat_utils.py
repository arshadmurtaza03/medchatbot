from langchain_groq import ChatGroq

API_KEY = None
MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7


def get_chat_model(api_key: str = None) -> ChatGroq:
    return ChatGroq(
        api_key=api_key or API_KEY,
        model=MODEL,
        temperature=TEMPERATURE,
    )


def ask_chat_model(chat_model: ChatGroq, prompt: str) -> str:
    response = chat_model.invoke(prompt)
    return response.content
