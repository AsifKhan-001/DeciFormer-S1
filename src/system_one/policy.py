import numpy as np


class DecisionPolicy:
    """Maps confidence / margin / entropy to ROUTE, CLARIFY, ABSTAIN or ESCALATE."""

    def __init__(self, route_threshold=0.85, clarify_threshold=0.5, min_margin=0.15,
                 max_entropy=1.0, ood_threshold=0.0):
        self.route, self.clarify, self.min_margin = route_threshold, clarify_threshold, min_margin
        self.max_entropy, self.ood = max_entropy, ood_threshold

    def __call__(self, confidence, margin, entropy=None, ood_score=None) -> list:
        c = np.atleast_1d(np.asarray(confidence, dtype=float))
        m = np.atleast_1d(np.asarray(margin, dtype=float))
        action = np.full(c.shape, "ABSTAIN", dtype="<U9")
        action[c >= self.clarify] = "CLARIFY"
        route = (c >= self.route) & (m >= self.min_margin)
        if entropy is not None:
            route &= np.atleast_1d(np.asarray(entropy, dtype=float)) <= self.max_entropy
        action[route] = "ROUTE"
        if ood_score is not None and self.ood > 0:
            action[np.atleast_1d(np.asarray(ood_score, dtype=float)) > self.ood] = "ESCALATE"
        return action.tolist()
