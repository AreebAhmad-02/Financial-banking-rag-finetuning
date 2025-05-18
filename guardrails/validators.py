# guardrails/validators.py

from guardrails.hub import DetectPII, DetectJailbreak, NSFWText, ToxicLanguage

# Input validators: block prompt injections and PII
input_validators = [
    DetectJailbreak(on_fail="block"),
    DetectPII(on_fail="block"),
]

# Output validators: block disallowed content or redact
output_validators = [
    NSFWText(on_fail="block"),
    ToxicLanguage(on_fail="block"),
    DetectPII(on_fail="redact"),
]
