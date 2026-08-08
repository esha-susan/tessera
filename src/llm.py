
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq")


def call_llm(prompt: str, system: str = "") -> str:
    """
    Sends a prompt to whichever provider is configured in .env.
    Returns the raw text response.
    """
    if PROVIDER == "groq":
        return _call_groq(prompt, system)
    elif PROVIDER == "gemini":
        return _call_gemini(prompt, system)
    elif PROVIDER == "claude":
        return _call_claude(prompt, system)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


def _call_groq(prompt: str, system: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, system: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system or None,
    )
    response = model.generate_content(prompt)
    return response.text


def _call_claude(prompt: str, system: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    # quick manual test: python3 -m src.llm
    reply = call_llm(
        prompt="Reply with exactly the word: pong",
        system="You are a terse test responder.",
    )
    print(f"[provider={PROVIDER}] response: {reply}")
