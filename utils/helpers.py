"""
AutoStream Agent — Utility Helpers

Provides validation and data-checking utilities used across the agent.
"""

import re


def is_valid_email(email: str) -> bool:
    """
    Validate email format using a standard regex pattern.

    Args:
        email: The email string to validate.

    Returns:
        True if the email matches a valid format, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def is_lead_data_complete(lead_data: dict) -> bool:
    """
    Check if all required lead fields have been collected.

    Required fields: name, email, platform.

    Args:
        lead_data: Dictionary with collected lead fields.

    Returns:
        True if all three fields are non-empty strings.
    """
    required_fields = ["name", "email", "platform"]
    for field in required_fields:
        value = lead_data.get(field, "")
        if not value or not str(value).strip():
            return False
    return True


def get_missing_fields(lead_data: dict) -> list[str]:
    """
    Identify which lead fields are still missing.

    Args:
        lead_data: Dictionary with collected lead fields.

    Returns:
        List of missing field names.
    """
    required_fields = ["name", "email", "platform"]
    missing = []
    for field in required_fields:
        value = lead_data.get(field, "")
        if not value or not str(value).strip():
            missing.append(field)
    return missing


def format_lead_summary(lead_data: dict) -> str:
    """
    Format collected lead data into a readable summary string.

    Args:
        lead_data: Dictionary with lead fields.

    Returns:
        Formatted string summarizing the lead.
    """
    return (
        f"Name: {lead_data.get('name', 'N/A')}\n"
        f"Email: {lead_data.get('email', 'N/A')}\n"
        f"Platform: {lead_data.get('platform', 'N/A')}"
    )
