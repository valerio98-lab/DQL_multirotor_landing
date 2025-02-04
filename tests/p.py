from typing import Any, Optional

try:
    LOGGING = 1
    from loguru import logger
except ImportError as e:
    print(e)
    logger: Optional[Any] = None
    LOGGING = 0

print(LOGGING)

if LOGGING:
    assert logger is not None  # Helps Pylance infer the correct type
    logger.debug("Ciao")
