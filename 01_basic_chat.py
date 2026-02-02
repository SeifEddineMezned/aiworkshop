import ollama

MODEL = "llama3.1:8b"  # change if needed

print("🤖 Basic Chat (no memory)")
print("Type 'exit' to quit.")

while True:
    user = input("You: ").strip()
    if user.lower() == "exit":
        print("Goodbye!")
        break

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": user},
    ]

    try:
        resp = ollama.chat(model=MODEL, messages=messages)
        print("Assistant:", resp["message"]["content"], " ")
    except Exception as e:
        print("❌ Error talking to Ollama:", e)

        print("🔧 Fix: make sure Ollama is running:  ollama serve")
        print("🔧 Fix: make sure the model exists:  ollama list")