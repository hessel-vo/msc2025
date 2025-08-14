import re
from typing import List, Tuple


EMAIL = (
    re.compile(r"""(^|[\b\s@,?!;:)(\'".<\[\]])([^\b\s@?!;,:)(’\"<]+@[^\b\s@!?;,/]*[^\b\s@?!;,/:)(’\">.]\.\w{2,})(?=$|[\b\s@,?!;:)(’'".>\[\]])"""),
    r'\1[EMAIL_ADDRESS]'
)

IPV4 = (
    re.compile(r'(^|[\b\s@?,!;:’\"(.)])((?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3})($|[\s@,?!;:’\"(.)])'),
    '[IPV4]'
)

IPV6 = (
    re.compile(r'(^|[\b\s@?,!;:’\"(.)])((?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9]))($|[\s@,?!;:’\"(.)])'),
    '[IPV6]'
)

SECRET = (
    re.compile(r'\b(api_key|secret|token|password|auth_token|bearer)\s*[:=]\s*["\']?([A-Za-z0-9+/_-]{16,})["\']?', re.IGNORECASE),
    r'\1:[SECRET]'
)


ALL_PATTERNS = [EMAIL, IPV4, IPV6, SECRET]

def redact_pii(text: str, patterns: List[Tuple[re.Pattern, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Iterates through PII patterns, replaces any matches, and reports which patterns were found
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
                sensitive_data = match.group(2)
                
                found_instances.append((log_name, sensitive_data))
                
                return match.expand(replacement)

            text = pattern.sub(standard_replacer, text)

    return text, found_instances