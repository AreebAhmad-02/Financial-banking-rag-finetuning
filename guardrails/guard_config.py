# guardrails/guard_config.py

from guardrails import Guard
from .validators import input_validators, output_validators

# Create separate Guard instances for input and output
input_guard = Guard()
output_guard = Guard()

# Attach validators
for validator in input_validators:
    input_guard += validator

for validator in output_validators:
    output_guard += validator
