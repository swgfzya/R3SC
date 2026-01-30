from collections import Counter
from scipy import integrate
from typing import List, Dict

class BetaStoppingCriteria:
    def __init__(self, conf_thresh: float = 0.95):
        self.conf_thresh = conf_thresh

    def should_stop(self, answers: List[str]) -> Dict:
        if not answers:
            return {'most_common': None, 'prob': 0, 'stop': False}

        counts = Counter(answers).most_common(2)
        if len(counts) == 1:
            a, b = counts[0][1], 0
            most_common_ans = counts[0][0]
        else:
            a, b = counts[0][1], counts[1][1]
            most_common_ans = counts[0][0]

        a = float(a)
        b = float(b)
        if a + b == 0:
            return {'most_common': most_common_ans, 'prob': 0, 'stop': False}

        try:
            numerator = integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0.5, 1)[0]
            denominator = integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0, 1)[0]
            prob = numerator / denominator if denominator != 0 else 0
        except Exception as e:
            print(f"积分错误: {e}")
            prob = 0

        return {
            'most_common': most_common_ans,
            'prob': prob,
            'stop': prob >= self.conf_thresh
        }