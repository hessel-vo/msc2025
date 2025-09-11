import re
from typing import List, Tuple

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

# Specific email address that does not need to be redacted
EXCEPTION = "git@github.com"

# Regex pattern to find email addresses.
EMAIL_PATTERN = re.compile(r"""(^|[\b\s@,?!;:)(\'".<\[\]])([^\b\s@?!;,:)(’\"<]+@[^\b\s@!?;,/]*[^\b\s@?!;,/:)(’\">.]\.\w{2,})(?=$|[\b\s@,?!;:)(’'".>\[\]])""")
# Replacement string. \1 is a backreference to the first capturing group (e.g., a space).
EMAIL_REPLACEMENT = r'\1[EMAIL_ADDRESS]'
LOG_NAME = '[EMAIL_ADDRESS]'


def redact_emails(text: str) -> Tuple[str, List[Tuple[str, str]]]:

    found_instances: List[Tuple[str, str]] = []

    def replacer(match: re.Match) -> str:
        # Group 2 of the regex captures the actual email address.
        email = match.group(2)

        # Check for the specific exception string.
        if EXCEPTION in email.lower():
            return match.group(0) # Return the original full match (do not redact).

        # Check if the TLD is in the list.
        try:
            domain = email.split('@')[1]
            tld = domain.split('.')[-1]
            if tld.lower() not in TLD:
                return match.group(0)
        except IndexError:
            return match.group(0) 

        # If the checks pass, redact the email.
        found_instances.append((LOG_NAME, email))
        
        return match.expand(EMAIL_REPLACEMENT)

    redacted_text = EMAIL_PATTERN.sub(replacer, text)

    return redacted_text, found_instances