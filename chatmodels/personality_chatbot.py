# chatbot.py — A CLI chatbot with personality modes and memory
# -------------------------------------------------------------
# This chatbot lets the user pick a mood/personality for the AI.
# It keeps track of the full conversation (chat history) so the model
# remembers what was said earlier in the session.
#
# Key concepts shown here:
#   - SystemMessage  → sets the AI's personality
#   - HumanMessage   → the user's input
#   - AIMessage      → the model's reply
#   - Chat history   → a list of all three, sent with every request

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)

# --- Choose a personality ---
print("\n🤖 Welcome! Choose an AI personality:")
print("  1 → Angry (grumpy and short-tempered)")
print("  2 → Funny (jokes and laughs)")
print("  3 → Wise  (calm, thoughtful, gives advice)")

choice = input("\nEnter 1, 2 or 3: ").strip()

if choice == "1":
    system_prompt = "You are a grumpy, impatient AI. You respond with frustration and sarcasm."
elif choice == "2":
    system_prompt = "You are a hilarious AI comedian. Every response includes a joke or pun."
else:
    system_prompt = "You are a wise, calm AI mentor. You give thoughtful, balanced advice."

# --- Start with the system message in history ---
# This is the "memory" — starts with one SystemMessage, grows with each turn
chat_history = [
    SystemMessage(content=system_prompt)
]

print("\n✅ Chatbot ready! Type your message. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit", "0"]:
        print("\nBot: Goodbye! 👋")
        break

    # Add the user's message to history
    chat_history.append(HumanMessage(content=user_input))

    # Send the entire history to the model (this is how memory works)
    response = model.invoke(chat_history)

    # Add the AI's reply to history so the next turn has full context
    chat_history.append(AIMessage(content=response.content))

    print(f"\nBot: {response.content}\n")

# Optional: show the full conversation at the end
print("\n--- Full conversation history ---")
for msg in chat_history:
    role = type(msg).__name__
    print(f"[{role}]: {msg.content[:80]}...")
