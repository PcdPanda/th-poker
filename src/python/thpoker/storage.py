"""Append-only JSONL session logs: one record per line, each carrying `schema_version`."""

from   dataclasses              import asdict, fields
import json
from   pathlib                  import Path
import threading
from   typing                   import Any, ClassVar, TypeVar

SCHEMA_VERSION = 1
R = TypeVar("R", bound="Record")


class SessionLog:
    """Writes one session's records. The file is opened per write, so a crash loses at most
    the record being written."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()  # background reviews append from their own thread
        self.deleted = False

    def append(self, record_type: str, payload: dict[str, Any]):
        record = {"schema_version": SCHEMA_VERSION, "type": record_type, **payload}
        with self.lock:
            if self.deleted:
                return
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def delete(self):
        """Remove the log; a later append, such as a review finishing afterwards, is dropped
        rather than starting the file again."""
        with self.lock:
            self.deleted = True
            self.path.unlink(missing_ok=True)


def read_session_log(path: Path) -> list[dict[str, Any]]:
    """All records of a log, in order. Raises `ValueError` for a record from a newer schema."""
    return read_records(path, 0)[0]


def read_records(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """The complete records after byte `offset` and the offset after them: a line still being
    written is left for the next read. Raises `ValueError` like `read_session_log`."""
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    complete = data[: data.rfind(b"\n") + 1]
    records = []
    for number, line in enumerate(complete.decode("utf-8").splitlines(), 1):
        record = json.loads(line)
        if not isinstance(record, dict) or "type" not in record:
            raise ValueError(f"{path}: record {number} after byte {offset} is not a log record")
        if record.get("schema_version", 0) > SCHEMA_VERSION:
            raise ValueError(
                f"{path}: record {number} after byte {offset} has schema version "
                f"{record['schema_version']}, newer than supported {SCHEMA_VERSION}"
            )
        records.append(record)
    return records, offset + len(complete)


def default_log_dir() -> Path:
    return Path.home() / ".poker_trainer" / "sessions"


def default_training_log() -> Path:
    """Where predict-then-reveal estimates are kept, across sessions."""
    return Path.home() / ".poker_trainer" / "training.jsonl"


def default_decisions_log() -> Path:
    """Where reviewed decisions are kept for leak statistics, across sessions."""
    return Path.home() / ".poker_trainer" / "decisions.jsonl"


class Record:
    """A dataclass kept in a log: its fields in order, read back ignoring the log's own keys."""

    __dataclass_fields__: ClassVar[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[R], data: dict[str, Any]) -> R:
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    @classmethod
    def read(cls: type[R], path: Path, kind: str) -> list[R]:
        """Every record of `kind` in the log at `path`; none if there is no log yet."""
        if not path.exists():
            return []
        return [cls.from_dict(r) for r in read_session_log(path) if r["type"] == kind]
