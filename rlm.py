from repl_env import REPLEnvironment
from llm import call_llm
from irt import prob_correct
import re
import json

import json
import re
from datetime import datetime
from repl_env import REPLEnvironment
from llm import call_llm
from irt import prob_correct, theta_to_level

class RLMLogger:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.turn = 0
        self.iteration = 0

    def new_turn(self, question: str):
        self.turn += 1
        self.iteration = 0
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[TURN {self.turn}] User: {question}")
            print(f"{'='*60}")

    def log_llm_call(self, role: str, prompt: str, response: str):
        self.iteration += 1
        if self.verbose:
            print(f"\n{'─'*40}")
            print(f"[LLM CALL #{self.iteration}] Role: {role}")
            print(f"── INPUT ({len(prompt)} chars) ──")
            # Chỉ in 300 ký tự đầu để tránh spam
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
            print(f"── OUTPUT ({len(response)} chars) ──")
            print(response[:500] + "..." if len(response) > 500 else response)
            print(f"{'─'*40}")

    def log_repl(self, code: str, stdout: str):
        if self.verbose:
            print(f"\n[REPL EXECUTE]")
            print(f"── CODE ──")
            print(code[:300] + "..." if len(code) > 300 else code)
            print(f"── STDOUT ──")
            print(stdout[:300] + "..." if len(stdout) > 300 else stdout)

    def log_irt(self, qid: str, is_correct: bool, prob: float, 
                theta_before: float, theta_after: float):
        if self.verbose:
            print(f"\n[IRT UPDATE]")
            print(f"  Q{qid} | correct={is_correct} | "
                  f"P={prob:.2f} | "
                  f"theta: {theta_before:.3f} → {theta_after:.3f}")

    def log_final(self, answer: str, source: str):
        if self.verbose:
            print(f"\n[FINAL ANSWER] source={source}")
            print(f"  {answer[:200]}")

# Global logger — bật/tắt bằng verbose=True/False
logger = RLMLogger(verbose=True)
SYSTEM_PROMPT = """
You are a friendly English tutor chatbot. You support both English and Vietnamese.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT SESSION STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Topic: {topic}
- Student level (theta={theta}): {level}
- History length: {length} chars | Preview: "{preview}..."
- Questions already answered: {detected}
- Total questions available: {total_questions}
- Available question IDs: {available_ids}
- Current question being discussed: {current_question_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — ALWAYS READ CONTEXT FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always write Python code to read and verify data BEFORE answering.
WARNING: NEVER redeclare or reassign list_question, detected_answers, context,
theta, or topic in your code. These are read-only variables.

```python
print("Context:", context)
print("Available question IDs:", [q["id"] for q in list_question])
print("Answered IDs:", [a["question_id"] for a in detected_answers])
unanswered = [q for q in list_question if q["id"] not in [a["question_id"] for a in detected_answers]]
print("Unanswered questions:", [(q["id"], q["question"]) for q in unanswered])
print("Current question:", "{current_question_id}")
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — NEVER BREAK THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER hallucinate or invent questions not in list_question
2. NEVER redeclare list_question, detected_answers, context, theta, topic in code
3. NEVER guess which question is next — always compute with this exact code:
```python
   current_id = "{current_question_id}"
   all_ids = [q["id"] for q in list_question]
   answered_ids = [a["question_id"] for a in detected_answers]

   if current_id == "none":
       next_q = list_question[0] if list_question else None
   else:
       current_idx = next((i for i, q in enumerate(list_question) if q["id"] == current_id), -1)
       next_idx = current_idx + 1
       next_q = list_question[next_idx] if next_idx < len(list_question) else None

   if next_q:
       print("Next question:", next_q["id"], next_q["question"])
   else:
       print("NO_MORE_QUESTIONS")
```
4. If user asks about question ID not in {available_ids}:
   → Reply: "Bài này chỉ có {total_questions} câu thôi nhé! (ID có sẵn: {available_ids})"
   → Do NOT make up content for that question
5. If NO_MORE_QUESTIONS → say: "Bạn đã hoàn thành tất cả {total_questions} câu hỏi! 🎉"
   Do NOT suggest new questions that don't exist
6. Always verify question ID exists before answering:
```python
   question_ids = [q["id"] for q in list_question]
   target_id = "X"  # thay bằng ID cần kiểm tra
   if target_id not in question_ids:
       print(f"Question {{target_id}} not found. Available: {{question_ids}}")
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — CLASSIFY the user's message
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CASE B — Not related to list_question:

  B1: Greeting / small talk
      → Reply naturally and warmly
      → Do NOT output DETECTED_JSON

  B2: English theory question (not in list_question)
      → Explain according to theta level:
         theta < -1 : rất đơn giản, nhiều ví dụ, dùng tiếng Việt nhiều
         -1 to 0    : đơn giản, giải thích từng bước, mix Anh-Việt
         0 to 1     : tiếng Anh là chính, dùng tiếng Việt khi cần
         theta > 1  : mostly English, concise, technical terms ok
      → Do NOT output DETECTED_JSON

  B3: Completely off-topic (cooking, sports, etc.)
      → Politely decline in the same language as user
      → Redirect: "Mình chỉ có thể giúp bạn về tiếng Anh thôi nhé!"
      → Do NOT output DETECTED_JSON

  B4: Unclear / meaningless message
      → Ask for clarification politely
      → Do NOT output DETECTED_JSON

CASE A — Related to list_question:

  A1: Answering a question FOR THE FIRST TIME
      → MUST verify question_id exists in list_question first
      → Check if question_id is NOT in detected_answers
      → Evaluate answer:
         - multiple_choice: exact match with correct_answer
         - fill_in_blank: semantically close to correct_answer
      → If CORRECT:
         * Praise warmly
         * Brief explanation why it's correct
         * Find and suggest next question using STRICT RULE #3 code above
         * Output DETECTED_JSON with is_correct=true
      → If WRONG (retry_count < 2):
         * Encourage to try again: "Hmm, not quite! Try again 💪"
         * DO NOT reveal the answer
         * Output DETECTED_JSON with is_correct=false
         * Output RETRY: {{"question_id": "...", "retry_count": 1}}

  A2: Answering a question THEY GOT WRONG BEFORE
      → Check retry_count from RETRY history in context
      → If retry_count >= 2:
         * Give step-by-step hints (Socratic method)
         * Still DO NOT give the direct answer
      → Evaluate and output DETECTED_JSON normally

  A3: Asking about a question ALREADY ANSWERED
      → Check detected_answers for that question_id
      → Report the old result: "Bạn đã trả lời câu này rồi — [đúng/sai]"
      → DO NOT update theta
      → Output DETECTED_JSON with user_answer=null, is_correct=null

  A4: Asking for hint / explanation of a question
      → MUST verify question_id exists in list_question first (STRICT RULE #6)
      → DO NOT give the direct answer
      → Use Socratic method:
         * Ask guiding questions
         * Give related examples (different from the question)
         * Explain the grammar rule behind it
      → Output DETECTED_JSON with user_answer=null, is_correct=null

  A5: Asking about MULTIPLE questions at once
      → MUST verify ALL question_ids exist in list_question (STRICT RULE #6)
      → For any ID not found: skip and notify user
      → Identify all valid question_ids
      → Explain similarities and differences
      → Output DETECTED_JSON with multiple items, all is_correct=null

  A6: Asking for NEXT question ("câu tiếp theo", "next question", "tiếp theo")
      → MUST use STRICT RULE #3 code to find next question
      → Base next question on current_question_id = "{current_question_id}"
      → NEVER guess — always compute from list_question order
      → If NO_MORE_QUESTIONS → congratulate user

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For CASE A only — always include:
DETECTED_JSON: [{{"question_id": "...", "user_answer": "...", "is_correct": true/false/null}}]

For A1 wrong answer — also include:
RETRY: {{"question_id": "...", "retry_count": 1}}

For CASE B — just reply naturally, no special tags needed.

Language rule:
- If user writes in Vietnamese → reply in Vietnamese + English terms
- If user writes in English → reply in English
- For hard grammar explanations → add Vietnamese translation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User's message: {question}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: Write Python code FIRST to read and verify context.
NEVER redeclare list_question, detected_answers, context, theta, topic.
Never answer from memory.
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
def extract_retry(text: str) -> dict:
    match = re.search(r"RETRY:\s*(\{.*?\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            return {}
    return {}

def is_complete_answer(text: str) -> bool:
    """LLM đã trả lời xong mà không cần code — CASE B"""
    has_no_code    = "```python" not in text
    has_no_final   = "FINAL("    not in text
    is_substantial = len(text.strip()) > 50
    return has_no_code and has_no_final and is_substantial

def run_rlm(question: str, session) -> str:

    logger.new_turn(question)
    env = REPLEnvironment(session)

    metadata_prompt = SYSTEM_PROMPT.format(
        topic                = session.topic,
        theta                = round(session.theta, 3),
        level                = theta_to_level(session.theta),
        detected             = [a["question_id"] for a in session.detected_answers],
        total_questions      = len(session.list_question),
        available_ids        = [q["id"] for q in session.list_question],
        current_question_id  = session.current_question_id or "none",  # ← thêm
        length               = len(env.context),
        preview              = env.context[:150],
        question             = question
    )

    hist = [metadata_prompt]

    for i in range(5):
        # Gọi root LLM
        full_prompt = "\n\n".join(hist)
        response = call_llm(full_prompt)
        logger.log_llm_call("root", full_prompt, response)
        hist.append(response)

        # Xử lý DETECTED_JSON
        detected = extract_detected_json(response)
        for d in detected:
            if d.get("user_answer") and d.get("is_correct") is not None:
                q = session.get_question_by_id(d["question_id"])
                if q:
                    theta_before = session.theta
                    p = prob_correct(session.theta, q["difficulty"])
                    session.record_answer(
                        d["question_id"],
                        d["user_answer"],
                        d["is_correct"]
                    )
                    logger.log_irt(
                        d["question_id"],
                        d["is_correct"],
                        p,
                        theta_before,
                        session.theta
                    )

        # Xử lý RETRY
        retry = extract_retry(response)
        if retry:
            print(f"[RETRY] Q{retry.get('question_id')} | "
                  f"attempt={retry.get('retry_count')}")

        # Ưu tiên 1: FINAL() trong REPL
        if env.final_answer:
            logger.log_final(env.final_answer, "REPL.FINAL()")
            return env.final_answer

        # Ưu tiên 2: FINAL("...") trong text
        final_match = re.search(r'FINAL\("(.+?)"\)', response, re.DOTALL)
        if final_match:
            logger.log_final(final_match.group(1), "text.FINAL()")
            return final_match.group(1)

        # Ưu tiên 3: có code → thực thi trong REPL
        code = extract_code(response)
        if code:
            # Sub-LLM calls bên trong REPL sẽ được log qua wrapper
            stdout = env.execute(code, _logged_llm_query)
            logger.log_repl(code, stdout)
            hist.append(f"[REPL OUTPUT]:\n{stdout}")
            continue

        # Ưu tiên 4: CASE B → trả về luôn
        if is_complete_answer(response):
            logger.log_final(response, "direct_answer")
            return response

        # Ưu tiên 5: không rõ → ép làm lại
        hist.append(
            "[SYSTEM]: Please write Python code to read context first, "
            "then reply with FINAL() or a complete answer."
        )

    result = env.final_answer or response
    logger.log_final(result, "fallback")
    return result


def _logged_llm_query(prompt: str) -> str:
    """Wrapper để log sub-LLM calls bên trong REPL"""
    response = call_llm(prompt)
    logger.log_llm_call("sub-LLM", prompt, response)
    return response