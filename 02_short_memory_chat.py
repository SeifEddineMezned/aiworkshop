import ollama

MODEL = "llama3.1:8b"  
MAX_TURNS = 16        

SYSTEM_PROMPT = """You are a helpful personal assistant.
Rules:
- Be friendly and clear
- Be concise
- Ask at most ONE clarification question if needed
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("🤖 Chat with Short-Term Memory (chat history)")
print("Type 'exit' to quit.")

def trim_history(msgs, max_turns: int):
    system = [m for m in msgs if m["role"] == "system"][:1]
    rest = [m for m in msgs if m["role"] != "system"]
    return system + rest[-max_turns:]

while True:
    user = input("You: ").strip()
    if user.lower() == "exit":
        print("Goodbye!")
        break

    messages.append({"role": "user", "content": user})
    messages = trim_history(messages, MAX_TURNS)

    try:
        resp = ollama.chat(model=MODEL, messages=messages)
        assistant = resp["message"]["content"]
        messages.append({"role": "assistant", "content": assistant})
        print("Assistant:", assistant, "")


    except Exception as e:
        print("❌ Error talking to Ollama:", e)
        print("🔧 Fix: make sure Ollama is running:  ollama serve")
        print("🔧 Fix: make sure the model exists:  ollama list")