import re

def remove_markdown_wrapping(code_string):
    """
    Removes the markdown code block wrapping from a string.

    Args:
        code_string: The string containing the code with markdown wrapping.

    Returns:
        The code without the markdown wrapping, or the original string if no wrapping is found.
    """
    pattern = r"^\s*```(?:\w+)?\n(.*?)\n```\s*$"
    match = re.search(pattern, code_string, re.DOTALL)
    if match:
        return match.group(1)
    return code_string

# Example Usage
wrapped_code_python = """```python
def hello_world():
    print("Hello, World!")
```"""

wrapped_code_java = """```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```"""

no_language_code = """```
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```"""

unwrapped_code = """
print("This code is not wrapped.")
test = 0
print(test)
"""


unwrapped_python = remove_markdown_wrapping(wrapped_code_python)
unwrapped_java = remove_markdown_wrapping(wrapped_code_java)
unwrapped_c = remove_markdown_wrapping(no_language_code)
unwrapped_plain = remove_markdown_wrapping(unwrapped_code)


print("--- Python Example ---")
print(unwrapped_python)
print("\n--- Java Example ---")
print(unwrapped_java)
print("\n--- C++ Example (No Language Identifier) ---")
print(unwrapped_c)
print("\n--- Unwrapped Code Example ---")
print(unwrapped_plain)