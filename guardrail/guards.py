from guardrails.hub import DetectJailbreak
from guardrails import Guard

# Setup Guard
input_guard = Guard().use(
    DetectJailbreak
)

# for validator in output_validators:
#     output_guard += validator

# Example usage:
# validated_input = input_guard.validate(user_input)
# validated_output = output_guard.validate(llm_output)