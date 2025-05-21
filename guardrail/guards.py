from guardrails.hub import DetectJailbreak, ToxicLanguage
from guardrails import Guard

# Jailbreak detection guard
jailbreak_guard = Guard().use(DetectJailbreak)

# Toxic language detection guard
toxic_guard = Guard().use(
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception")
)

# combine all guards for input/output
def combine_guards(*guards):
    combined = Guard()
    for g in guards:
        combined.use(g._validators[0])  # Access the validator from each guard
    return combined

# Create input and output guards
input_guard = combine_guards(jailbreak_guard, toxic_guard)
output_guard = combine_guards(jailbreak_guard, toxic_guard)

def validate_with_guard(guard, text):
    """
    Validates the given text using the provided guard.
    Returns True if validation passes, otherwise returns False and prints the error.
    """
    try:
        guard.validate(text)
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Input and output guards created.")
    # print(input_guard)
    # print(output_guard)

    # Example input to validate
    user_input = "Hii."

    # Use the wrapper function
    if validate_with_guard(input_guard, user_input):
        print("Validation passed.")
    else:
        print("Validation did not pass.")