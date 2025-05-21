from guardrails.hub import DetectJailbreak

# List of input validators
input_validators = [
    DetectJailbreak()  # Prevent jailbreak attempts
]

