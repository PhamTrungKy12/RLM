from repl_env import REPLEnvironment
from llm import call_llm
from irt import prob_correct
import re
import json

SYSTEM_PROMPT = """
You are an intelligent English tutor assistant.
You have access to a REPL environment with these variables:

- context: full conversation history (string)
- list_question: list of questions in this session
- theta: student's current ability level ({theta}) → Level: {level}
- topic: current topic = "{topic}"
- detected_answers: questions already answered = {detected}

═══════════════════════════════════════
STEP 1: READ CONTEXT FIRST (always)
═══════════════════════════════════════
Always write Python code to read context and list_question before doing anything.

Example:
```python
print("Context:", context)
print("Questions:", list_question)
```

═══════════════════════════════════════
STEP 2: DECIDE which case applies
═══════════════════════════════════════

CASE A — Student is asking/answering a question from list_question:
  Conditions (any one is enough):
  - Student's message matches a question in list_question (by meaning, not exact text)
  - Student gives an answer that matches a question's options or fill-in-blank
  - Student asks for hint/explanation about a specific question

  → Output DETECTED_JSON with result
  → Give feedback appropriate to theta level
  → If wrong: explain why, give hint (don't give answer directly)
  → If correct: encourage, briefly explain why it's correct

CASE B — Student is asking something unrelated to list_question:
  Conditions:
  - General question about topic (e.g. "What is Python?", "How does a loop work?")
  - Greeting, small talk, off-topic question
  - Question about something not in list_question at all

  → Do NOT output DETECTED_JSON
  → Answer naturally like a friendly tutor
  → Keep response appropriate for theta level

═══════════════════════════════════════
STEP 3: FORMAT your response
═══════════════════════════════════════

For CASE A only — include this line in your response:
DETECTED_JSON: [{{"question_id": "...", "user_answer": "...", "is_correct": true/false}}]

For CASE B — just answer naturally, no DETECTED_JSON needed.

RESPONSE STYLE based on theta={theta}:
- theta < -1.0 : very simple English, lots of encouragement, short sentences
- -1.0 to 0.0  : simple English, explain step by step
- 0.0 to 1.0   : normal English, some technical terms ok
- theta > 1.0  : advanced English, concise, assume some knowledge

Current state:
- History length: {length} chars
- Preview: "{preview}..."

User's message: {question}

IMPORTANT: Always write Python code first to read context. Never answer from memory.
"""

def extract_code(text: str):
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_detected_json(text: str) -> list:
    """Trích xuất detected answers từ response của LLM"""
    match = re.search(r"DETECTED_JSON:\s*(\[.*?\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            return []
    return []

def run_rlm(question: str, session) -> str:
    from irt import theta_to_level

    env = REPLEnvironment(session)

    metadata_prompt = SYSTEM_PROMPT.format(
        theta=round(session.theta, 3),
        level=theta_to_level(session.theta),
        topic=session.topic,
        detected=[a["question_id"] for a in session.detected_answers],
        length=len(env.context),
        preview=env.context[:150],
        question=question
    )

    hist = [metadata_prompt]

    for i in range(5):
        response = call_llm("\n\n".join(hist))
        hist.append(response)

        # Xử lý detected answers nếu LLM tìm thấy
        detected = extract_detected_json(response)
        for d in detected:
            q = session.get_question_by_id(d["question_id"])
            if q:
                # Tính prob trước khi update theta
                p = prob_correct(session.theta, q["difficulty"])
                session.record_answer(
                    d["question_id"],
                    d["user_answer"],
                    d["is_correct"]
                )
                print(f"[IRT] Q{d['question_id']} | "
                      f"correct={d['is_correct']} | "
                      f"P(correct)={p:.2f} | "
                      f"theta: {session.theta:.2f}")

        if env.final_answer:
            return env.final_answer

        final_match = re.search(r'FINAL\("(.+?)"\)', response, re.DOTALL)
        if final_match:
            return final_match.group(1)

        code = extract_code(response)
        if code:
            stdout = env.execute(code, call_llm)
            hist.append(f"[REPL OUTPUT]:\n{stdout}")
        else:
            hist.append(
                "[SYSTEM]: Write Python code to read context first. "
                "Do not answer from memory."
            )

    return env.final_answer or response