import io
import contextlib

class REPLEnvironment:
    def __init__(self, session):
        self.session = session
        self.context = self._history_to_string(session.history)
        self.variables = {}
        self.final_answer = None  # ← đảm bảo có field này

    def _history_to_string(self, history: list) -> str:
        if not history:
            return "[No conversation history yet]"
        lines = []
        for turn in history:
            lines.append(f"[{turn['role'].upper()}]: {turn['content']}")
        return "\n".join(lines)

    def _set_final(self, answer: str):  # ← thêm method này
        self.final_answer = answer

    def execute(self, code: str, llm_fn) -> str:
    # Lưu giá trị gốc của các biến protected
        protected = {
            "list_question": self.session.list_question,
            "detected_answers": self.session.detected_answers,
            "topic": self.session.topic,
            "theta": self.session.theta,
        }

        namespace = {
            "context": self.context,
            "llm_query": llm_fn,
            "FINAL": self._set_final,
            **protected
        }

        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, namespace)

            # Chỉ lưu biến MỚI, không cho override biến protected
            for k, v in namespace.items():
                if not k.startswith("_") and k not in protected and k not in (
                    "context", "llm_query", "FINAL"
                ):
                    self.variables[k] = v

            # Restore lại biến protected phòng trường hợp bị override
            for k, v in protected.items():
                namespace[k] = v

        except Exception as e:
            return f"[REPL ERROR]: {e}"

        return stdout_capture.getvalue()