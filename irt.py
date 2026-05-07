import math

A = 1.0  # discrimination cố định

def prob_correct(theta: float, b: float) -> float:
    """
    Công thức 2PL với a=1:
    P = 1 / (1 + e^(-(theta - b)))
    """
    return 1.0 / (1.0 + math.exp(-(theta - b)))

def update_theta(theta: float, is_correct: bool, b: float, 
                 learning_rate: float = 0.3) -> float:
    """
    Cập nhật theta đơn giản sau mỗi câu trả lời:
    - Đúng: theta tăng dựa trên độ khó câu hỏi
    - Sai:  theta giảm dựa trên độ khó câu hỏi
    """
    p = prob_correct(theta, b)
    
    if is_correct:
        # Đúng câu khó → tăng nhiều, đúng câu dễ → tăng ít
        delta = learning_rate * (1 - p)
    else:
        # Sai câu dễ → giảm nhiều, sai câu khó → giảm ít
        delta = -learning_rate * p
    
    # Giới hạn theta trong khoảng [-3, 3]
    return max(-3.0, min(3.0, theta + delta))

def theta_to_level(theta: float) -> str:
    """Chuyển theta thành mô tả trình độ"""
    if theta < -1.5:
        return "Beginner"
    elif theta < -0.5:
        return "Elementary"
    elif theta < 0.5:
        return "Intermediate"
    elif theta < 1.5:
        return "Upper-Intermediate"
    else:
        return "Advanced"