try:
    from loguru import logger  # type: ignore
except ImportError:

    class LoggerStub:
        """Fallback logger that mimics loguru's API when loguru is not installed."""

        @staticmethod
        def debug(_message: str) -> None: ...
        @staticmethod
        def info(_message: str) -> None: ...
        @staticmethod
        def success(_message: str) -> None: ...
        @staticmethod
        def warning(_message: str) -> None: ...
        @staticmethod
        def error(_message: str) -> None: ...
        @staticmethod
        def critical(_message: str) -> None: ...

    logger = LoggerStub()
