#!/usr/bin/env python3
"""
HR AI Assistant Chatbot — API 1 (OpenAI / ChatGPT)
Uses API1 key from .env file.
"""

import os
import sys

# ─────────────────────────────────────────────
# LOAD .env FILE
# ─────────────────────────────────────────────

def load_env(filepath=".env"):
    """Manually load .env file without needing python-dotenv."""
    if not os.path.exists(filepath):
        return
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(key, value)

load_env()

# ─────────────────────────────────────────────
# HR SYSTEM PROMPT
# ─────────────────────────────────────────────

HR_SYSTEM_PROMPT = """You are a helpful HR Assistant for a company. You help employees and HR staff with:
- Leave & attendance policies
- Payroll and salary queries
- Onboarding and offboarding processes
- Employee benefits and perks
- Performance review processes
- Workplace policies and code of conduct
- Recruitment and job postings
- Training and development programs
- Grievance handling

Be professional, empathetic, and concise. If you don't know a specific company policy,
ask the user to check with their HR department directly.
"""

# ─────────────────────────────────────────────
# INSTALL PACKAGES IF NEEDED
# ─────────────────────────────────────────────

def install_packages():
    import subprocess
    print("📦 Checking/installing required packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "openai", "-q"],
        capture_output=True
    )
    print("✅ Done!\n")

# ─────────────────────────────────────────────
# OPENAI CHAT FUNCTION (API1)
# ─────────────────────────────────────────────

def chat_openai(api_key: str, history: list, user_input: str):
    """
    Send a message using OpenAI ChatCompletion API (API1 key).

    Args:
        api_key   : OpenAI API key (API1 from .env)
        history   : List of message dicts [{"role": ..., "content": ...}, ...]
        user_input: The user's latest message

    Returns:
        (reply: str, history: list)  on success
        (None,  error_msg: str)      on failure
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Inject system prompt on first turn
        if not history:
            history.append({"role": "system", "content": HR_SYSTEM_PROMPT})

        history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",      # cost-effective, fast
            messages=history,
            temperature=0.7,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})
        return reply, history

    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────
# MAIN CHATBOT LOOP
# ─────────────────────────────────────────────

def run_chatbot():
    install_packages()

    api_key = os.environ.get("HR_chatbot_API", "").strip()

    if not api_key:
        print("\n❌ API1 key not found in .env file.")
        print("   Please set:  API1 = sk-proj-...")
        sys.exit(1)

    masked = api_key[:8] + "..." + api_key[-4:]

    print("\n" + "═" * 55)
    print("       🤖  HR AI Assistant  —  API 1 (OpenAI)")
    print("═" * 55)
    print(f"   🔑 Key : {masked}")
    print(f"   🧠 Model: gpt-4o-mini")
    print("═" * 55)
    print("   Type your question below.")
    print("   Commands:  'quit' to exit  |  'clear' to reset chat")
    print("═" * 55 + "\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("\n👋 Goodbye! Have a great day.")
            break
        if user_input.lower() == "clear":
            history = []
            print("🔄 Chat history cleared.\n")
            continue

        print("\nHR Assistant: ", end="", flush=True)

        reply, result = chat_openai(api_key, history, user_input)

        if reply is None:
            print(f"\n❌ Error: {result}")
            print("💡 Tip: Check that API1 is valid and has billing enabled.\n")
            continue

        history = result
        print(reply)
        print()


if __name__ == "__main__":
    run_chatbot()
