from __future__ import annotations

import logging
import uuid


LOGGER = logging.getLogger("pricepredict")


def user_error(exc: Exception, context: str) -> tuple[str, str]:
    """Log technical detail and return a safe, actionable message plus incident ID."""
    incident = uuid.uuid4().hex[:8].upper()
    LOGGER.exception("%s failed [incident=%s]", context, incident, exc_info=exc)
    text = str(exc).lower()
    if "no models are eligible" in text:
        message = "No models fit the current dataset and run mode. Choose Balanced mode or review the selected model families."
    elif "numeric" in text or "could not" in text and "convert" in text:
        message = "One or more values cannot be interpreted as numeric measurements. Review the highlighted data fields and try again."
    elif "missing" in text and "column" in text:
        message = "A field required by the selected model is missing. Restore it or allow the saved preprocessing rules to handle it."
    elif "memory" in text or "allocate" in text:
        message = "This experiment exceeds the available resource budget. Use Quick mode or reduce the dataset size."
    elif "target" in text:
        message = "The selected outcome cannot be used for training. Confirm that it is numeric and contains enough historical values."
    else:
        message = f"{context} could not be completed safely. Review the dataset warnings and try again."
    return message, incident
