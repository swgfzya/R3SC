import re


def _strip_string(s):
    """
    Helper function for stripping certain characters from a string.

    This function is used to clean up the answer string extracted from the model's
    output, removing LaTeX commands, spaces, and other artifacts.
    """
    # Remove \text{...}
    s = re.sub(r'\\text\{[^}]*\}', '', s)

    # Remove LaTeX commands
    s = s.replace("\\", "")

    # Remove braces and dollar signs
    s = s.replace("{", "").replace("}", "")
    s = s.replace("$", "")

    # Remove leading/trailing whitespace
    s = s.strip()

    return s


def delete_extra_zero(n):
    """
    Removes trailing '.0' from a number string if it's an integer.
    e.g., "123.0" -> "123"
    """
    try:
        # Check if the string contains a decimal point and if it represents a whole number
        if '.' in n and float(n) == int(float(n)):
            return str(int(float(n)))
    except (ValueError, TypeError):
        # If conversion fails, return the original string
        pass
    return n


def is_equiv(str1, str2, verbose=False):
    """
    Compares two strings for equivalence, specifically for math answers.
    It handles fractions, percentages, and numerical comparisons.
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    str1 = str(str1).strip()
    str2 = str(str2).strip()

    # Handling percentages
    if str1.endswith("%"):
        str1 = str1[:-1]
        try:
            str1 = str(float(str1) / 100)
        except ValueError:
            return False

    if str2.endswith("%"):
        str2 = str2[:-1]
        try:
            str2 = str(float(str2) / 100)
        except ValueError:
            return False

    # Handling fractions
    def get_val(s):
        if "/" in s:
            try:
                parts = s.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
        try:
            return float(s)
        except ValueError:
            return None

    val1 = get_val(str1)
    val2 = get_val(str2)

    if val1 is not None and val2 is not None:
        # Using a small tolerance for float comparison
        return abs(val1 - val2) < 1e-6

    # If not numerical, perform a direct string comparison
    return str1 == str2