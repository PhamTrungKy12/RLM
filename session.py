from irt import prob_correct, update_theta, theta_to_level
from typing import Optional

class Session:
    def __init__(self, list_question: list, topic: str, theta: float = 0.0):
        self.list_question = list_question
        self.topic = topic
        self.theta = theta
        self.history = []
        self.detected_answers = []  # tất cả câu đã trả lời

    def get_question_by_id(self, qid: str) -> Optional[dict]:
        for q in self.list_question:
            if q["id"] == qid:
                return q
        return None

    def record_answer(self, question_id: str, user_answer: str, 
                      is_correct: bool):
        """Lưu kết quả và cập nhật theta"""
        q = self.get_question_by_id(question_id)
        if not q:
            return

        b = q["difficulty"]
        p = prob_correct(self.theta, b)

        self.detected_answers.append({
            "question_id": question_id,
            "user_answer": user_answer,
            "is_correct": is_correct,
            "prob_correct": round(p, 3),
            "theta_before": round(self.theta, 3),
        })

        # Cập nhật theta
        self.theta = update_theta(self.theta, is_correct, b)

    def get_state_summary(self) -> str:
        """Tóm tắt state hiện tại để inject vào REPL"""
        answered_ids = [a["question_id"] for a in self.detected_answers]
        
        lines = [
            f"Topic: {self.topic}",
            f"Theta: {round(self.theta, 3)} ({theta_to_level(self.theta)})",
            f"Questions answered so far: {answered_ids}",
            f"Answer history: {self.detected_answers}"
        ]
        return "\n".join(lines)