"""
AutoStream Agent — Tool Definitions

Defines the mock_lead_capture tool that is invoked ONLY after
all lead data (name, email, platform) has been collected and validated.
"""

from utils.helpers import is_valid_email


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulate capturing a qualified lead in the CRM system.

    This function is called ONLY when:
    1. All three fields (name, email, platform) are collected
    2. Email has been validated
    3. Intent has been classified as high_intent

    Args:
        name:     Full name of the lead.
        email:    Valid email address.
        platform: Target platform (e.g., YouTube, TikTok).

    Returns:
        Confirmation message string.

    Raises:
        ValueError: If any input fails validation.
    """
    # --- Input Validation ---
    if not name or not name.strip():
        raise ValueError("Name cannot be empty.")

    if not email or not is_valid_email(email):
        raise ValueError(f"Invalid email address: {email}")

    if not platform or not platform.strip():
        raise ValueError("Platform cannot be empty.")

    # --- Sanitize inputs ---
    name = name.strip()
    email = email.strip().lower()
    platform = platform.strip()

    # --- Simulate CRM capture ---
    print(f"\n{'='*50}")
    print(f"  [OK] LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Name:     {name}")
    print(f"  Email:    {email}")
    print(f"  Platform: {platform}")
    print(f"{'='*50}\n")

    return f"Lead captured successfully: {name}, {email}, {platform}"
