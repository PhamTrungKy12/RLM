import io
import contextlib

class REPLEnvironment:
    def __init__(self, session):
        self.session = session
        self.context = self._history_to_string(session.history)
        self.variables = {}
        self.final_answer = None

    def _history_to_string(self, history: list) -> str:
        if not history:
            return "[No conversation history yet]"
        lines = []
        for turn in history:
            lines.append(f"[{turn['role'].upper()}]: {turn['content']}")
        return "\n".join(lines)

    def execute(self, code: str, llm_fn) -> str:
        namespace = {
            "context": self.context,
            "list_question": self.session.list_question,
            "theta": self.session.theta,
            "topic": self.session.topic,
            "detected_answers": self.session.detected_answers,
            "llm_query": llm_fn,
            "FINAL": self._set_final,
            **self.variables
        }

        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, namespace)

            for k, v in namespace.items():
                if not k.startswith("_") and k not in (
                    "context", "list_question", "theta",
                    "topic", "detected_answers", "llm_query", "FINAL"
                ):
                    self.variables[k] = v

        except Exception as e:
            return f"[REPL ERROR]: {e}"

        return stdout_capture.getvalue()

    def _set_final(self, answer: str):
        self.final_answer = answer