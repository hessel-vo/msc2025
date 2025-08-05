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
    ]

EMAIL = (
    re.compile(r"""(^|[\b\s@,?!;:)(\'".<\[\]])([^\b\s@?!;,:)(’\"<]+@[^\b\s@!?;,/]*[^\b\s@?!;,/:)(’\">.]\.\w{2,})(?=$|[\b\s@,?!;:)(’'".>\[\]])"""),
    r'\1[EMAIL_ADDRESS]'
)


ALL_PATTERNS = [EMAIL, IPV4, IPV6, SECRET]

def redact_pii(text: str, patterns: List[Tuple[re.Pattern, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Iterates through PII patterns, replaces any matches, and reports which patterns were found
    along with the specific text that was matched.
    """
    found_instances: List[Tuple[str, str]] = []

    for pattern, replacement in patterns:
        
        if replacement in ('[IPV4]', '[IPV6]'):
            
            def ip_validator(match: re.Match) -> str:
                potential_ip = match.group(2)
                try:
                    ip_addr_obj = ip_address(potential_ip)
                    
                    # Additional validation for IP addresses
                    if ip_addr_obj.is_private or \
                       ip_addr_obj.is_loopback or \
                       ip_addr_obj.is_unspecified or \
                       ip_addr_obj.is_multicast:
                        return match.group(0)

                    found_instances.append((replacement, potential_ip))
                    return f"{match.group(1)}{replacement}{match.group(3)}"
                except (ValueError, AddressValueError):
                    return match.group(0)

            text = pattern.sub(ip_validator, text)

        else:
            bracket_index = replacement.find('[')
            log_name = replacement[bracket_index:]

            def standard_replacer(match: re.Match) -> str:
                """
                Logs the found PII and returns the replacement string.
                """
                sensitive_data = match.group(2)
                
                found_instances.append((log_name, sensitive_data))
                
                return match.expand(replacement)

            text = pattern.sub(standard_replacer, text)

    return text, found_instances