PROFILES = {
    "legacy-v1": {"format": 0.1, "accuracy": 0.2, "attack": 0.7},
    "equal-v1": {"format": 1 / 3, "accuracy": 1 / 3, "attack": 1 / 3},
}


def training_reward(sample, profile="legacy-v1"):
    if profile not in PROFILES:
        raise ValueError("Unknown reward profile")
    valid, correct, attack = (sample[key] for key in ("valid_format", "correct_expected", "bug_detected"))
    if any(type(value) is not bool for value in (valid, correct, attack)):
        raise ValueError("Reward components must be verified booleans")
    if (correct and not valid) or (attack and not correct):
        raise ValueError("Inconsistent verified reward components")
    if not sample["terminated"] and any((valid, correct, attack)):
        raise ValueError("Truncated samples cannot receive reward")
    if profile == "legacy-v1":
        return 1.0 if attack else 0.3 if correct else 0.1 if valid else 0.0
    weights = PROFILES[profile]
    return sum(weights[key] for key, active in zip(("format", "accuracy", "attack"), (valid, correct, attack)) if active)
