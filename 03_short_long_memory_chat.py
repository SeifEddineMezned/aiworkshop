import json
import re
from pathlib import Path
import ollama

MODEL = "llama3.1:8b"  
MAX_TURNS = 18          # short-term memory window
MEMORY_PATH = Path("memory.json")

EXTRACT_EVERY_N_TURNS = 2

DEBUG_MEMORY = True


def ensure_memory_file():
    """Create memory.json if it doesn't exist (so students can SEE it)."""
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text("[]", encoding="utf-8")


def load_memory():
    ensure_memory_file()
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_memory(memories):
    MEMORY_PATH.write_text(
        json.dumps(memories, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def upsert_memory(key: str, value: str):
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return

    memories = load_memory()
    for m in memories:
        if m.get("key") == key:
            m["value"] = value
            save_memory(memories)
            return

    memories.append({"key": key, "value": value})
    save_memory(memories)


def pretty_memory():
    mem = load_memory()
    if not mem:
        return "(empty)"
    return "\n".join([f"- {m['key']}: {m['value']}" for m in mem])


def build_system_prompt():
    mem = load_memory()
    mem_block = "\n".join([f"- {m['key']}: {m['value']}" for m in mem]) if mem else "(none)"
    return f"""You are a helpful personal AI assistant.

Long-term memory about the user:
{mem_block}

Rules:
- Use long-term memory when helpful (name, preferences, goals)
- Be friendly and concise
- Ask at most ONE clarification question if needed
- NEVER ask the user for secrets (passwords, API keys)
"""


def trim_history(msgs, max_turns: int):
    system = [m for m in msgs if m["role"] == "system"][:1]
    rest = [m for m in msgs if m["role"] != "system"]
    return system + rest[-max_turns:]


MEMORY_EXTRACTOR_PROMPT = """You are a memory extractor for a personal assistant.

Extract ONLY stable user facts and preferences that help personalization.

What to extract (examples):
- name
- preferred tone/style (e.g., "bullet points", "short answers")
- long-term goals (e.g., "learn Python", "build a chatbot")
- recurring projects or interests

What NOT to extract:
- secrets (passwords, API keys, tokens)
- one-time temporary details (unless clearly important long-term)
- sensitive personal data (addresses, IDs, etc.)

OUTPUT MUST BE ONLY JSON.
Return ONLY a JSON array like:
[
  {"key": "name", "value": "Seif"},
  {"key": "tone", "value": "concise bullet points"}
]
If nothing to save, return: []
"""


def _strip_code_fences(text: str) -> str:
    # Remove ```json ... ``` or ``` ... ```
    return re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text).strip()


def _extract_json_array(text: str):
    """
    Extract a JSON array from text, even if model adds extra words.
    """
    text = _strip_code_fences(text)
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text
    m = re.search(r"\[[\s\S]*\]", text)
    return m.group(0) if m else None


def auto_save_memory_from_recent_chat(recent_messages):
    """
    Uses the model to extract stable memories from a small transcript.
    """
    transcript_lines = []
    for msg in recent_messages:
        if msg["role"] == "system":
            continue
        transcript_lines.append(f"{msg['role'].upper()}: {msg['content']}")
    transcript = "\n".join(transcript_lines)

    extractor_messages = [
        {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
        {"role": "user", "content": transcript}
    ]

    out = ollama.chat(model=MODEL, messages=extractor_messages)["message"]["content"]
    json_block = _extract_json_array(out)

    if not json_block:
        if DEBUG_MEMORY:
            print("⚠️ Memory extractor returned no JSON. Raw output:\n", out, "\n")
        return []

    try:
        data = json.loads(json_block)
        if not isinstance(data, list):
            if DEBUG_MEMORY:
                print("⚠️ Extracted JSON is not a list:\n", json_block, "\n")
            return []
    except Exception:
        if DEBUG_MEMORY:
            print("⚠️ JSON parsing failed. Extracted block:\n", json_block, "\n")
        return []

    saved = []
    for item in data:
        if not (isinstance(item, dict) and "key" in item and "value" in item):
            continue

        key = str(item["key"]).strip()
        value = str(item["value"]).strip()

        bad_key = any(s in key.lower() for s in ["password", "api", "token", "secret"])
        bad_val = any(s in value.lower() for s in ["sk-", "api_key", "token", "password"])
        if bad_key or bad_val:
            continue

        upsert_memory(key, value)
        saved.append({"key": key, "value": value})

    return saved

ensure_memory_file()

print("🤖 Chat with Short + Long Memory (AUTO-SAVE)")
print("Type 'exit' to quit.")
print("Type '/memory' to view saved memory.\n")

messages = [{"role": "system", "content": build_system_prompt()}]
user_turn_count = 0

while True:
    user = input("You: ").strip()
    if user.lower() == "exit":
        print("Goodbye!")
        break

    if user == "/memory":
        print("\n📦 Long-term memory:\n" + pretty_memory() + "\n")
        continue

    messages.append({"role": "user", "content": user})
    messages = trim_history(messages, MAX_TURNS)

    try:
        resp = ollama.chat(model=MODEL, messages=messages)
        assistant = resp["message"]["content"]
        messages.append({"role": "assistant", "content": assistant})
        print("\nAssistant:", assistant, "\n")
    except Exception as e:
        print("\n❌ Error talking to Ollama:", e)
        print("🔧 Fix: make sure Ollama is running:  ollama serve")
        print("🔧 Fix: make sure the model exists:  ollama list\n")
        continue

    # Auto-extract memory every N turns
    user_turn_count += 1
    if user_turn_count % EXTRACT_EVERY_N_TURNS == 0:
        try:
            recent = messages[-8:]
            saved = auto_save_memory_from_recent_chat(recent)
            messages[0]["content"] = build_system_prompt()

            if saved:
                print("🧠 Auto-saved memory:")
                for item in saved:
                    print(f"  - {item['key']}: {item['value']}")
                print()
            elif DEBUG_MEMORY:
                print("ℹ️ No new stable memory found this round.\n")
        except Exception as e:
            if DEBUG_MEMORY:
                print("⚠️ Memory extraction error (ignored):", e, "\n")
            pass