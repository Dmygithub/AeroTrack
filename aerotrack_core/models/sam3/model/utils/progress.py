import os


_FALSE_VALUES = {"0", "false", "no", "off", ""}


def sam3_progress_enabled(default: bool = True) -> bool:
    """Return True when SAM3 progress bars should be shown.

    Controlled by env var `SAM3_SHOW_PROGRESS`:
      - unset -> `default`
      - "0/false/no/off" (case-insensitive) -> False
      - anything else -> True
    """
    value = os.getenv("SAM3_SHOW_PROGRESS")
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_VALUES


def should_disable_tqdm(disable: bool = False, default: bool = True) -> bool:
    """Combine local disable flags with the global SAM3 progress setting."""
    if disable:
        return True
    return not sam3_progress_enabled(default=default)
