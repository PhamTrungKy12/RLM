from rlm import run_rlm
from session import Session

# ===== CONFIG =====
list_question = [
    {
        "id": "1",
        "question": "Look! He (run)………………",
        "type": "fill_in_blank",
        "answer": "run",
        "correct_answer": "is running",
        "difficulty": 3.777280928294443,
    },
    {
        "id": "2",
        "question": "I (not wear)……………… a coat as it isn’t cold.",
        "type": "fill_in_blank",
        "answer": "not wear",
        "correct_answer": "am not wearing",
        "difficulty": 4.850750087409718,
    },
    {
        "id": "3",
        "question": "Why (you/sit)……………… at my desk?",
        "type": "fill_in_blank",
        "answer": "you/sit",
        "correct_answer": "are you sitting",
        "difficulty": 4.721223542044355,
    },
    {
        "id": "4",
        "question": "What (the baby/do)……………… at the moment?",
        "type": "fill_in_blank",
        "answer": "the baby/do",
        "correct_answer": "is the baby doing",
        "difficulty": 4.146295705710651,
    },
    {
        "id": "5",
        "question": "I (read)……………… a play by Shaw.",
        "type": "fill_in_blank",
        "answer": "read",
        "correct_answer": "am reading",
        "difficulty": 4.958567835319735,
    },
    {
        "id": "6",
        "question": "He (teach)……………… French and (learn)……………… Greek.",
        "type": "fill_in_blank",
        "answer": "teach, learn",
        "correct_answer": "is teaching, is learning",
        "difficulty": 4.850750087409718,
    },
    {
        "id": "7",
        "question": "She (knit)……………… and (listen)……………… to the radio.",
        "type": "fill_in_blank",
        "answer": "knit, listen",
        "correct_answer": "is knitting, is listening",
        "difficulty": 5.381847266621481,
    },
    {
        "id": "8",
        "question": "I (meet)……………… Peter tonight. He (take)……………… me to the theatre.",
        "type": "fill_in_blank",
        "answer": "meet, take",
        "correct_answer": "am meeting, is taking",
        "difficulty": 4.801315426922857,
    },
    {
        "id": "9",
        "question": "She (always/lose)………… her keys.",
        "type": "fill_in_blank",
        "answer": "always/lose",
        "correct_answer": "is always losing",
        "difficulty": 4.747122707436239,
    },
    {
        "id": "10",
        "question": "He (always/come)………… to work late.",
        "type": "fill_in_blank",
        "answer": "always/come",
        "correct_answer": "is always coming",
        "difficulty": 4.149960608250699,
    },
    {
        "id": "11",
        "question": "Generally, to describe a noun, we don't use more than 2 or 3 Adjectives.",
        "type": "multiple_choice",
        "answer": ["True", "False"],
        "correct_answer": "True",
        "difficulty": 5.978397283971757,
    },
    {
        "id": "12",
        "question": "My dad is ___ man.",
        "type": "multiple_choice",
        "answer": ["an old sweet", "a sweetest", "a sweet old"],
        "correct_answer": "a sweet old",
        "difficulty": 9.0,
    },
    {
        "id": "13",
        "question": "Jenny had a ___ in her hair yesterday.",
        "type": "multiple_choice",
        "answer": ["nice pink bow", "pink nice bow", "bow nice pink"],
        "correct_answer": "nice pink bow",
        "difficulty": 9.0,
    },
    {
        "id": "14",
        "question": "My hair is long and ___.",
        "type": "multiple_choice",
        "answer": ["curly", "smart", "slim"],
        "correct_answer": "curly",
        "difficulty": 9.0,
    },
    {
        "id": "15",
        "question": "I have three balls. This red ball is the ___.",
        "type": "multiple_choice",
        "answer": ["littlest", "smallest", "most little"],
        "correct_answer": "smallest",
        "difficulty": 9.0,
    },
    {
        "id": "16",
        "question": "There is not an established order to be respected when we use many Adjectives in a sentence.",
        "type": "multiple_choice",
        "answer": ["True", "False"],
        "correct_answer": "False",
        "difficulty": 5.8958544494459915,
    },
    {
        "id": "17",
        "question": "They grew up in ___ house in Peru.",
        "type": "multiple_choice",
        "answer": [
            "a comfortable, little",
            "a little, comfortable",
            "a comfortable little"
        ],
        "correct_answer": "a comfortable little",
        "difficulty": 9.0,
    },
    {
        "id": "18",
        "question": "She lost a ___ .",
        "type": "multiple_choice",
        "answer": ["small black cat", "cat small black", "black small cat"],
        "correct_answer": "small black cat",
        "difficulty": 9.0,
    },
    {
        "id": "19",
        "question": "Peter drives a bright yellow sports car. It's very ___.",
        "type": "multiple_choice",
        "answer": ["wild", "short", "fast"],
        "correct_answer": "fast",
        "difficulty": 9.0,
    },
    {
        "id": "20",
        "question": "Give me the ___ cup.",
        "type": "multiple_choice",
        "answer": ["plastic big green", "big green plastic", "big plastic green"],
        "correct_answer": "big green plastic",
        "difficulty": 9.0,
    }
]

session = Session(
    list_question=list_question,
    topic="Present Continuous",
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