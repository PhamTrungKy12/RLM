from rlm import run_rlm
from session import Session

# ===== CONFIG =====
list_question = [
    {
        "id": "1",
        "question": "What is the output of print(2**3)?",
        "type": "multiple_choice",
        "answer": ["6", "8", "9", "None"],
        "correct_answer": "8",
        "difficulty": 0.5,
    },
    {
        "id": "2",
        "question": "Python is ___ typed language.",
        "type": "fill_in_blank",
        "answer": "dynamic",
        "correct_answer": "dynamically",
        "difficulty": -0.3,
    }
]

session = Session(
    list_question=list_question,
    topic="Python Basics",
    theta=0.0
)

print("RLM Tutor Chatbot (type 'quit' to exit)\n")
print(f"Topic: {session.topic} | Initial theta: {session.theta}\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break

    session.history.append({"role": "user", "content": user_input})

    answer = run_rlm(user_input, session)
    print(f"Bot: {answer}\n")

    session.history.append({"role": "assistant", "content": answer})

    # In state sau mỗi turn
    print(f"[State] theta={session.theta:.3f} | "
          f"answered={[a['question_id'] for a in session.detected_answers]}\n")