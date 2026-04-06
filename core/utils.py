"""Shared utility functions for Dr. Data core modules."""


def remove_tableau_brackets(name: str) -> str:
    """Safely remove surrounding [] from Tableau field names.

    strip("[]") is dangerous -- it strips ALL leading/trailing [ and ] chars,
    mangling names like "[Sales [Net]]" into "Sales [Net".
    This function only removes the outermost balanced bracket pair.
    """
    if name and name.startswith("[") and name.endswith("]"):
        return name[1:-1]
    return name
