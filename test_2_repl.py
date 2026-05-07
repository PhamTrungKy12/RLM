from repl_env import REPLEnvironment
from llm import call_llm

# Tạo history giả
history = [
    {"role": "user", "content": "My name is Nam"},
    {"role": "assistant", "content": "Nice to meet you Nam!"},
    {"role": "user", "content": "I am 25 years old"},
    {"role": "assistant", "content": "Got it, you are 25!"},
]

env = REPLEnvironment(history)

# Kiểm tra context có đúng không
print("=== Context ===")
print(env.context)
print()

# Test thực thi code đơn giản
print("=== Test execute code ===")
code = """
chunk = context[:50]
print("First 50 chars:", chunk)
"""
output = env.execute(code, call_llm)
print("Output:", output)