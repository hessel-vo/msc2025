import re
from typing import List, Tuple

# List of Top-Level Domains (TLDs) that should be considered for redaction.
# Emails with TLDs not in this list will be ignored.
TLD = [ 
    "bmw",
    "com",
    "de",
    "io",
    "me",
    "mobi",
    "net",
    "org",
    "uk",
    "ibm",
    "fr",
    "global"
]

# A specific email address that should never be redacted, even if its TLD is in the list.
EXCEPTION = "git@github.com"

# Regex pattern to find potential email addresses.
EMAIL_PATTERN = re.compile(r"""(^|[\b\s@,?!;:)(\'".<\[\]])([^\b\s@?!;,:)(’\"<]+@[^\b\s@!?;,/]*[^\b\s@?!;,/:)(’\">.]\.\w{2,})(?=$|[\b\s@,?!;:)(’'".>\[\]])""")
# Replacement string. \1 is a backreference to the first capturing group (e.g., a space).
EMAIL_REPLACEMENT = r'\1[EMAIL_ADDRESS]'
# Name for logging purposes.
LOG_NAME = '[EMAIL_ADDRESS]'


def redact_emails(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Finds and replaces email addresses based on a TLD allowlist,
    while ignoring specific exceptions. It also reports which emails were redacted.

    Args:
        text: The input string to redact.

    Returns:
        A tuple containing:
        - The redacted text.
        - A list of tuples, where each tuple contains the redaction category 
          and the specific email that was found and redacted.
    """
    found_instances: List[Tuple[str, str]] = []

    def replacer(match: re.Match) -> str:
        """
        This function is called for each match found by re.sub.
        It decides whether a replacement should occur based on the new rules.
        """
        # Group 2 of the regex captures the actual email address.
        email = match.group(2)

        # 1. Check for the specific exception string.
        if EXCEPTION in email.lower():
            return match.group(0) # Return the original full match (do not redact).

        # 2. Check if the TLD is in the allowed list.
        # The TLD is the part of the domain after the last dot.
        try:
            domain = email.split('@')[1]
            tld = domain.split('.')[-1]
            if tld.lower() not in TLD:
                return match.group(0) # TLD not in list, do not redact.
        except IndexError:
            # This case handles malformed matches that don't have a '@' or '.'.
            return match.group(0) 

        # 3. If the checks pass, redact the email.
        found_instances.append((LOG_NAME, email))
        
        # Use match.expand() to handle the \1 backreference in the replacement string.
        return match.expand(EMAIL_REPLACEMENT)

    # Use re.sub with the replacer function to process the text.
    redacted_text = EMAIL_PATTERN.sub(replacer, text)

    return redacted_text, found_instances

# Example Usage:
if __name__ == '__main__':
    sample_text = (
        "Contact me at (valid@example.com) or [git@github.com for code. "
        "My work email is an.intern@bmw.de, but my private one is me@email.xyz. "
        "Reach out to [support@some-service.io] for help]."
    )
    redacted_text, findings = redact_emails(sample_text)
    
    print("Original Text:")
    print(sample_text)
    print("\n" + "="*30 + "\n")
    print("Redacted Text:")
    print(redacted_text)
    print("\n" + "="*30 + "\n")
    print("Found and Redacted PII Instances:")
    # Expected to find .com, .de, and .io, but ignore .xyz and git@github.com
    for category, data in findings:
        print(f"- {category}: {data}")