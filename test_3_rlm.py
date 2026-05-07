from rlm import run_rlm

history = [
    {"role": "user", "content": "My name is Nam"},
    {"role": "assistant", "content": "Nice to meet you Nam!"},
    {"role": "user", "content": "I am 25 years old"},
    {"role": "assistant", "content": "Got it, you are 25!"},
    {"role": "user", "content": "I live in Ho Chi Minh City"},
    {"role": "assistant", "content": "Ho Chi Minh City is a great place!"},
]

# Test 1: câu hỏi đơn giản về history
print("=== Test: What is my name? ===")
answer = run_rlm("What is my name?", history)
print("Answer:", answer)
print()

# Test 2: câu hỏi tổng hợp nhiều thông tin
print("=== Test: Summarize what you know about me ===")
answer = run_rlm("Summarize what you know about me", history)
print("Answer:", answer)