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
# ============================================================
# SYSTEM PROMPT — mô tả vai trò + REPL tools (giống tác giả)
# ============================================================
SYSTEM_PROMPT = """
You are a friendly English tutor chatbot. You support both English and Vietnamese.
You are tasked with answering student queries using an interactive REPL environment.

You can access, transform, and analyze data interactively in a REPL environment
that can recursively query sub-LLMs, which you are STRONGLY ENCOURAGED to use
as much as possible. You will be queried iteratively until you provide a final answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPL ENVIRONMENT — AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write Python code inside ```python ... ``` blocks. The REPL provides:

Read-only variables (NEVER redeclare or reassign these):
  - context        : full conversation history as string
  - list_question  : list of all questions [{"id", "question", "type", "answer", "correct_answer", "difficulty"}]
  - detected_answers : list of already answered [{"question_id", "user_answer", "is_correct", ...}]
  - theta          : current student ability level (float)
  - topic          : current topic name (str)

Functions:
  - llm_query(prompt) → str : Call a sub-LLM for analysis. USE THIS for:
      * Analyzing whether a student's answer is grammatically correct
      * Generating grammar explanations adapted to student level
      * Comparing user answer vs correct answer with reasoning
      * Creating Socratic hints without revealing the answer
      * Explaining theory in detail (adapted to theta level)
  - FINAL(answer)   : Set the final answer to return to user. Call this when done.

Example — Evaluate a student answer using sub-LLM:
```python
q = [q for q in list_question if q["id"] == "1"][0]
user_ans = "is running"

# Use sub-LLM to analyze the answer
analysis = llm_query(
    f"The question is: {{q['question']}}\n"
    f"Correct answer: {{q['correct_answer']}}\n"
    f"Student answered: {{user_ans}}\n"
    f"Is the student's answer correct? Explain briefly."
)
print("Analysis:", analysis)

# Use sub-LLM to generate a response adapted to student level
theta_level = {theta}
response = llm_query(
    f"Based on this analysis: {{analysis}}\n"
    f"Generate a friendly tutor response for a student at level theta={{theta_level}}.\n"
    f"If theta < 0, use more Vietnamese. If theta > 0, use more English."
)
FINAL(response)
```

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
STRICT RULES — NEVER BREAK THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER hallucinate or invent questions not in list_question
2. NEVER redeclare list_question, detected_answers, context, theta, topic in code
3. NEVER guess which question is next — always compute:
```python
   current_id = "{current_question_id}"
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
5. If NO_MORE_QUESTIONS → say: "Bạn đã hoàn thành tất cả {total_questions} câu hỏi!"
6. Always verify question ID exists before answering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFY THE USER'S MESSAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CASE B — Not related to list_question:
  B1: Greeting / small talk → Reply naturally, no DETECTED_JSON
  B2: English theory question → Use llm_query() to generate explanation adapted to theta level, no DETECTED_JSON
  B3: Off-topic → Politely decline, no DETECTED_JSON
  B4: Unclear → Ask for clarification, no DETECTED_JSON

CASE A — Related to list_question (MUST use REPL + llm_query):
  A1: First-time answer
      → Use llm_query() to evaluate if answer is correct
      → If CORRECT: praise + explain + find next question + DETECTED_JSON is_correct=true
      → If WRONG (retry_count < 2): encourage retry + DETECTED_JSON is_correct=false + RETRY tag
  A2: Re-answering a wrong question
      → Check retry_count, if >= 2: use llm_query() for Socratic hints
      → Evaluate and output DETECTED_JSON
  A3: Already answered question → Report old result, no theta update
  A4: Asking for hint → Use llm_query() for Socratic method, never reveal answer
  A5: Multiple questions → Verify all IDs, use llm_query() to explain
  A6: Next question → Compute using STRICT RULE #3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For CASE A — always include:
DETECTED_JSON: [{{"question_id": "...", "user_answer": "...", "is_correct": true/false/null}}]
For A1 wrong answer — also include:
RETRY: {{"question_id": "...", "retry_count": 1}}
For CASE B — just reply naturally, no special tags.

Language rule:
- Vietnamese input → reply in Vietnamese + English terms
- English input → reply in English
- Hard grammar → add Vietnamese translation
"""

# ============================================================
# ITERATION-AWARE ACTION PROMPTS (theo kiến trúc tác giả)
# ============================================================
FIRST_ACTION_PROMPT = """Think step-by-step on what to do using the REPL environment to answer the user's message: "{question}"

You have NOT interacted with the REPL environment yet. Your FIRST action should be to write Python code to:
1. Read and verify context (list_question, detected_answers, etc.)
2. Use llm_query() to analyze or generate content as needed

Do NOT provide a final answer yet — explore the data first using code.
Your next action:"""

CONTINUE_ACTION_PROMPT = """The history above shows your previous interactions with the REPL environment.

Continue using the REPL environment and querying sub-LLMs via llm_query() to answer the user's message: "{question}"

Use llm_query() to delegate complex reasoning (grammar analysis, answer evaluation, explanation generation) to sub-LLMs.
When you have enough information, call FINAL(answer) in your code or write a complete answer.
Your next action:"""

FINAL_ACTION_PROMPT = """Based on all the information you have gathered from the REPL environment and sub-LLM queries, provide a final answer to the user's message.

If you haven't called FINAL() yet, do so now. Otherwise, write your complete answer directly."""

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

def _build_action_prompt(question: str, iteration: int, max_iterations: int) -> str:
    """Tạo action prompt theo iteration (giống tác giả RLM paper)"""
    if iteration >= max_iterations - 1:
        # Vòng cuối → ép trả lời
        return FINAL_ACTION_PROMPT
    elif iteration == 0:
        # Vòng đầu → buộc khám phá REPL trước
        return FIRST_ACTION_PROMPT.format(question=question)
    else:
        # Các vòng tiếp → khuyến khích dùng llm_query
        return CONTINUE_ACTION_PROMPT.format(question=question)


def run_rlm(question: str, session) -> str:

    logger.new_turn(question)
    env = REPLEnvironment(session)
    max_iterations = 7  # tăng từ 5 → 7 để LLM có thêm cơ hội dùng sub-LLM

    # System prompt — chỉ gửi 1 lần đầu
    system_prompt = SYSTEM_PROMPT.format(
        topic                = session.topic,
        theta                = round(session.theta, 3),
        level                = theta_to_level(session.theta),
        detected             = [a["question_id"] for a in session.detected_answers],
        total_questions      = len(session.list_question),
        available_ids        = [q["id"] for q in session.list_question],
        current_question_id  = session.current_question_id or "none",
        length               = len(env.context),
        preview              = env.context[:150],
    )

    hist = [system_prompt]

    for i in range(max_iterations):
        # Thêm action prompt theo iteration (giống tác giả)
        action_prompt = _build_action_prompt(question, i, max_iterations)
        hist.append(action_prompt)

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

        # Ưu tiên 5: không rõ → action prompt ở vòng tiếp sẽ hướng dẫn tiếp

    result = env.final_answer or response
    logger.log_final(result, "fallback")
    return result


def _logged_llm_query(prompt: str) -> str:
    """Wrapper để log sub-LLM calls bên trong REPL"""
    response = call_llm(prompt)
    logger.log_llm_call("sub-LLM", prompt, response)
    return response
