from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import TypeDecorator


class UuidCompat(TypeDecorator):
    """Store string UUIDs portably while using native PostgreSQL UUID columns."""

    impl = String
    cache_ok = True

    def __init__(self, **kwargs):
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=False))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        return None if value is None else str(value)
