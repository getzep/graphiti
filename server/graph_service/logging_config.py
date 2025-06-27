import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry)


class GraphitiLogger:
    """Wrapper for structured logging with context support."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.request_id: Optional[str] = None

    def set_request_id(self, request_id: str):
        """Set request ID for correlation."""
        self.request_id = request_id

    def clear_request_id(self):
        """Clear request ID."""
        self.request_id = None

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with extra data."""
        extra_data = kwargs.copy()
        if self.request_id:
            extra_data['request_id'] = self.request_id

        record = self.logger.makeRecord(
            name=self.logger.name, level=level, fn='', lno=0, msg=message, args=(), exc_info=None
        )
        record.extra_data = extra_data
        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)


def setup_logging(log_level: str = 'INFO', log_format: str = 'structured'):
    """Setup logging configuration."""
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Set formatter based on format preference
    if log_format.lower() == 'structured':
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def generate_job_id() -> str:
    """Generate a unique job ID for tracking."""
    return str(uuid.uuid4())[:8]


def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return str(uuid.uuid4())[:8]


# Global logger instances
worker_logger = GraphitiLogger('graphiti.worker')
api_logger = GraphitiLogger('graphiti.api')
db_logger = GraphitiLogger('graphiti.db')
