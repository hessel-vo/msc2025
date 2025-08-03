"""
This module provides functionality for redacting PII and other sensitive data.
It now returns both the redacted text and a list of patterns that were found,
allowing for detailed logging of PII occurrences.
"""
import re
from typing import List, Tuple, Set

# =============================================================================
# == 1. DEFINE INDIVIDUAL PII PATTERNS AS NAMED CONSTANTS ==
# =============================================================================
# (This entire section of defining patterns like IPV4_SIMPLE, EMAIL, etc.
# remains identical to the previous version. It is omitted here for brevity
# but should be considered part of the final file.)
# --- Simple IP Address Patterns ---
IPV4_SIMPLE = (
    re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
    '[IP_ADDRESS_4S]'
)
IPV6_SIMPLE = (
    re.compile(r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))', re.IGNORECASE),
    '[IP_ADDRESS_6S]'
)

# --- Advanced IP Address Patterns (from GitHub source) ---
IPV4_ADVANCED = (
    re.compile(r'\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
    '[IP_ADDRESS_4A]'
)
IPV6_ADVANCED = (
    re.compile(r'\s*(?!.*::.*::)(?:(?!:)|:(?=:))(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)){6}(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)[0-9a-f]{0,4}(?:(?<=::)|(?<!:)|(?<=:)(?<!::):)|(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\s*', re.VERBOSE | re.IGNORECASE | re.DOTALL),
    '[IP_ADDRESS_6A]'
)

# --- Other Sensitive Data Patterns ---
EMAIL = (
    re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', re.IGNORECASE),
    '[EMAIL_ADDRESS]'
)
SECRETS = (
    re.compile(r'\b(api_key|secret|token|password|auth_token|bearer)\s*[:=]\s*["\']?([A-Za-z0-9+/_-]{16,})["\']?', re.IGNORECASE),
    r'\1:[REDACTED_SECRET]' # Use \1 to preserve the keyword
)
URL_WITH_CREDS = (
    re.compile(r'(https?|ftp|sftp|amqp)://[a-zA-Z0-9_.-]+:[^@\s]+@[a-zA-Z0-9_.-]+'),
    '[URL_WITH_CREDENTIALS]'
)
MAC_ADDRESS = (
    re.compile(r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'),
    '[MAC_ADDRESS]'
)
VIN = (
    re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
    '[VEHICLE_IDENTIFICATION_NUMBER]'
)


# =============================================================================
# == 2. COMPOSE PATTERN GROUPS FROM THE NAMED CONSTANTS ==
# =============================================================================

SIMPLE_IP_PATTERNS = [IPV4_SIMPLE, IPV6_SIMPLE]
ADVANCED_IP_PATTERNS = [IPV4_ADVANCED, IPV6_ADVANCED]
OTHER_SENSITIVE_PATTERNS = [EMAIL, SECRETS, URL_WITH_CREDS, MAC_ADDRESS, VIN]

ALL_PATTERNS_RECOMMENDED = OTHER_SENSITIVE_PATTERNS + ADVANCED_IP_PATTERNS


# =============================================================================
# == 3. REDACTION FUNCTION ==
# =============================================================================

# MODIFIED: The function signature and return type have been changed.
def redact_pii(text: str, patterns: List[Tuple[re.Pattern, str]]) -> Tuple[str, List[str]]:
    """
    Iterates through PII patterns, replaces any matches, and reports which
    patterns were found.

    Args:
        text: The input string to redact.
        patterns: A list of (regex_pattern, replacement_token) tuples.

    Returns:
        A tuple containing:
        - The redacted text (str).
        - A sorted list of the unique replacement tokens for patterns
          that were found in the text (List[str]).
    """
    found_patterns: Set[str] = set()

    for pattern, replacement in patterns:
        # Use re.subn() which returns a tuple of (new_string, number_of_subs_made)
        text, num_subs = pattern.subn(replacement, text)
        
        # If any substitutions were made, record the pattern's "name"
        if num_subs > 0:
            # For secrets, the replacement is complex, so we use a clear name.
            if "[REDACTED_SECRET]" in replacement:
                 found_patterns.add("[REDACTED_SECRET]")
            else:
                 found_patterns.add(replacement)

    # Return the modified text and a sorted list of the unique patterns found
    return text, sorted(list(found_patterns))

# =============================================================================
# == 4. TEST AND EVALUATION BLOCK ==
# =============================================================================

# MODIFIED: The test block is updated to reflect the new function signature.
if __name__ == '__main__':
    print("--- Testing PII Redaction Script ---")

    sample_text = """
    // User credentials and connection details
    // Contact: john.doe@example.com for support.
    const serverConfig = {
        host: "192.168.1.100", // Valid IP
        bad_ip: "300.0.0.1",
        port: 8080,
        ipv6_host: "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        api_key: "sk_live_1234567890abcdef12345678",
        password= "a-very-secure-password-that-is-long",
        db_url: "amqp://user:mysecretpassword@rabbitmq.host.com/vhost"
    };
    // Device and vehicle info
    let device_mac = "00-1A-2B-3C-4D-5E";
    let car_vin = "1GNEV923456789ABC";
    """

    print("\n" + "="*50)
    print("--- Original Text ---")
    print(sample_text)
    print("="*50)

    # --- Test Case 3: Recommended Final Configuration ---
    print("\n" + "="*50)
    print("--- Redaction using ALL_PATTERNS_RECOMMENDED ---")
    
    # MODIFIED: Unpack the new return tuple
    redacted_text, found_pii = redact_pii(sample_text, ALL_PATTERNS_RECOMMENDED)
    
    print("--- Redacted Text ---")
    print(redacted_text)
    
    # NEW: Print the list of patterns that were found
    print("\n--- Found PII Types ---")
    print(found_pii)
    print("="*50)

    print("\n--- Test Complete ---")