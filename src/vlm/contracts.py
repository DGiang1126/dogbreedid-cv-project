"""Shared schema 1.0 DTOs and deterministic validation for VLM Contracts 01–05.

Wire decoding verifies structure only. Binding functions take dependencies from
the caller's trusted scope. None of these functions grant approval or access.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from functools import lru_cache
import hashlib
import json
import math
import re
from types import MappingProxyType, UnionType
from typing import Annotated, Literal, Protocol, TypeVar, Union, get_args, get_origin, get_type_hints
import unicodedata
from urllib.parse import urlsplit


SCHEMA_VERSION = "1.0"
CLASS_IDS = tuple(range(30))
_EXPERIMENTS = frozenset({"E12-A", "E12-B", "E13-A", "E13-B", "E14-A", "E14-B", "E15"})
_RUN_ID = re.compile(
    r"(E12-[AB]|E13-[AB]|E14-[AB]|E15)_(pilot|formal|exploratory)_"
    r"([0-9]{8}T[0-9]{6}Z)_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    HASH_MISMATCH = "HASH_MISMATCH"
    MAPPING_MISMATCH = "MAPPING_MISMATCH"
    INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
    UNREVIEWED_SOURCE = "UNREVIEWED_SOURCE"
    TOKEN_TRUNCATION = "TOKEN_TRUNCATION"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    TIMEOUT = "TIMEOUT"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ContractError(ValueError):
    """Validation failure with a contract error code, without raw input text."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.INVALID_INPUT) -> None:
        super().__init__(message)
        self.code = code


def _require(condition: bool, message: str, code: ErrorCode = ErrorCode.INVALID_INPUT) -> None:
    if not condition:
        raise ContractError(message, code)


def _expect_record(value: object, record_type: type) -> None:
    _require(type(value) is record_type, f"expected validated {record_type.__name__} DTO")


def validate_class_id(value: object) -> int:
    _require(type(value) is int and 0 <= value < len(CLASS_IDS), "class_id must be an integer from 0 to 29")
    return value


def validate_sha256(value: object) -> str:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None, "expected 64 lowercase SHA-256 hex characters")
    return value


def verify_sha256(raw: bytes, expected_sha256: str) -> None:
    validate_sha256(expected_sha256)
    _require(type(raw) is bytes, "hash input must be bytes")
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "artifact SHA-256 mismatch", ErrorCode.HASH_MISMATCH)


def _commit_sha(value: str) -> None:
    _require(_COMMIT_SHA.fullmatch(value) is not None, "revision must be a full immutable commit SHA")


def _nonnegative(value: int | float) -> None:
    _require(value >= 0, "expected a nonnegative number")


def _positive(value: int) -> None:
    _require(value > 0, "expected a positive integer")


ClassId = Annotated[int, validate_class_id]
Sha256 = Annotated[str, validate_sha256]
_CommitSha = Annotated[str, _commit_sha]
_PositiveInt = Annotated[int, _positive]
_NonNegativeInt = Annotated[int, _nonnegative]
_Number = int | float
_NonNegativeNumber = Annotated[_Number, _nonnegative]
_Context = str | Literal[""]


def _utf8_text(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError("expected valid UTF-8 text") from exc


_RawText = Annotated[str, _utf8_text]


class _JsonValue:
    """Shape marker for arbitrary JSON values governed by a caller's rubric."""


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


def _unique(values: Collection[object], field: str) -> None:
    _require(len(set(values)) == len(values), f"{field} must be unique")


def _ordered_classes(records: tuple, field: str) -> None:
    _require(tuple(record.class_id for record in records) == CLASS_IDS, f"{field} must contain IDs 0–29 in order")


def _schema_version(value: str) -> None:
    _require(value == SCHEMA_VERSION, "unsupported schema_version", ErrorCode.UNSUPPORTED_SCHEMA)


_SchemaVersion = Annotated[str, _schema_version]


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def _parse_float(text: str) -> float:
    value = float(text)
    _require(math.isfinite(value), "JSON numbers must be finite")
    return value


def _reject_constant(_: str) -> None:
    raise ContractError("JSON numbers must be finite")


def decode_json(raw: bytes) -> object:
    """Read strict UTF-8 JSON before choosing the wire record to validate."""
    _require(type(raw) is bytes, "JSON input must be bytes")
    _require(not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM is not allowed")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_json_object,
            parse_float=_parse_float, parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ContractError("invalid UTF-8 JSON payload") from exc


@lru_cache(maxsize=None)
def _record_fields(record_type: type) -> dict:
    hints = get_type_hints(record_type, include_extras=True)
    return {field.name: hints[field.name] for field in fields(record_type)}


class _Record:
    __slots__ = ()

    def __post_init__(self) -> None:
        for name, annotation in _record_fields(type(self)).items():
            value = _field_value(getattr(self, name), annotation, name, wire=False)
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        """Cross-field rules for a record whose fields have passed shape validation."""


class _WireRecord(_Record):
    __slots__ = ()


def _json_value(value: object, name: str, *, wire: bool) -> object:
    if isinstance(value, Mapping):
        _require(not wire or type(value) is dict, f"{name} must be a JSON object")
        result = {}
        for key, item in value.items():
            _require(type(key) is str, f"{name} requires string object keys")
            _utf8_text(key)
            result[key] = _json_value(item, name, wire=wire)
        return MappingProxyType(result)
    if type(value) is list or (not wire and type(value) is tuple):
        return tuple(_json_value(item, name, wire=wire) for item in value)
    if type(value) is str:
        _utf8_text(value)
        return value
    if value is None or type(value) in (bool, int):
        return value
    _require(type(value) is float and math.isfinite(value), f"{name} must be a finite JSON value")
    return value


def _field_value(value: object, annotation: object, name: str, *, wire: bool) -> object:
    if annotation is _JsonValue:
        return _json_value(value, name, wire=wire)
    if annotation is _RawText:
        _require(type(value) is str, f"{name} must be str")
        _utf8_text(value)
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        checked = _field_value(value, args[0], name, wire=wire)
        for validate in args[1:]:
            validate(checked)
        return checked
    if origin in (UnionType, Union):
        if type(None) in args and value is None:
            return None
        if _Missing in args and value is MISSING and not wire:
            return MISSING
        choices = tuple(choice for choice in args if choice not in (type(None), _Missing))
        if len(choices) == 1:
            return _field_value(value, choices[0], name, wire=wire)
        for choice in choices:
            try:
                return _field_value(value, choice, name, wire=wire)
            except ContractError:
                continue
        raise ContractError(f"{name} has an invalid type or value")
    if origin is Literal:
        _require(any(type(value) is type(item) and value == item for item in args), f"{name} has an invalid value")
        return value
    if origin is tuple:
        array_types = (list,) if wire else (list, tuple)
        _require(type(value) in array_types, f"{name} must be an array")
        return tuple(_field_value(item, args[0], name, wire=wire) for item in value)
    if origin is Mapping:
        _require(isinstance(value, Mapping) and (not wire or type(value) is dict), f"{name} must be an object")
        key_type, item_type = args
        return MappingProxyType({
            _field_value(key, key_type, name, wire=wire): _field_value(item, item_type, name, wire=wire)
            for key, item in value.items()
        })
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if wire and type(value) is str:
            try:
                return annotation(value)
            except ValueError:
                pass
        _require(not wire and type(value) is annotation, f"{name} must use {annotation.__name__}")
        return value
    if isinstance(annotation, type) and issubclass(annotation, _Record):
        if wire:
            _require(issubclass(annotation, _WireRecord), f"{name} is Python-only")
            return _wire_record(annotation, value)
        _require(type(value) is annotation, f"{name} must be {annotation.__name__}")
        return value
    _require(type(value) is annotation, f"{name} must be {annotation.__name__}")
    if annotation is str:
        _require(bool(value.strip()), f"{name} must not be blank")
        _utf8_text(value)
    if annotation is float:
        _require(math.isfinite(value), f"{name} must be finite")
    return value


def _wire_record(record_type: type, value: object) -> _WireRecord:
    _require(type(value) is dict, f"{record_type.__name__} must be a JSON object")
    annotations = _record_fields(record_type)
    if "schema_version" in annotations and "schema_version" in value:
        _field_value(value["schema_version"], _SchemaVersion, "schema_version", wire=True)
    _require(not (value.keys() - annotations.keys()), f"{record_type.__name__} contains unknown fields")
    for field in fields(record_type):
        # Schema-version defaults are a Python convenience, never a wire default.
        if field.name not in value:
            _require(field.default is MISSING, f"{record_type.__name__} is missing {field.name}")
    return record_type(**{
        name: _field_value(item, annotations[name], name, wire=True)
        for name, item in value.items()
    })


_Wire = TypeVar("_Wire", bound=_WireRecord)


def decode_record(raw: bytes, record_type: type[_Wire]) -> _Wire:
    """Decode a JSON-only record; binding and publication policy are separate."""
    _require(isinstance(record_type, type) and issubclass(record_type, _WireRecord), "expected a JSON wire record type")
    return _wire_record(record_type, decode_json(raw))


def _wire_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _WireRecord):
        return {
            field.name: _wire_value(item)
            for field in fields(value)
            if (item := getattr(value, field.name)) is not MISSING
        }
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    if isinstance(value, Mapping):
        return {_wire_value(key): _wire_value(item) for key, item in value.items()}
    _require(not isinstance(value, _Record), "Python-only DTO has no JSON carrier")
    return value


def encode_record(record: _WireRecord) -> bytes:
    """Encode a validated JSON-only record, omitting absent optional fields."""
    _require(isinstance(record, _WireRecord), "expected a JSON wire record")
    return json.dumps(_wire_value(record), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalMapping(_Record):
    names: Mapping[ClassId, str]
    mapping_sha256: Sha256

    def _validate(self) -> None:
        _require(set(self.names) == set(CLASS_IDS), "canonical mapping must contain IDs 0–29", ErrorCode.MAPPING_MISMATCH)
        _unique(self.names.values(), "canonical names")


def decode_canonical_mapping(raw: bytes, *, expected_sha256: Sha256) -> CanonicalMapping:
    """Bind the existing ID-keyed metadata format to its trusted byte hash."""
    verify_sha256(raw, expected_sha256)
    names = decode_json(raw)
    _require(type(names) is dict and set(names) == {str(i) for i in CLASS_IDS},
             "canonical mapping keys must be strings 0–29", ErrorCode.MAPPING_MISMATCH)
    return CanonicalMapping(names={i: names[str(i)] for i in CLASS_IDS}, mapping_sha256=expected_sha256)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedRgb(_Record):
    """Decoded RGB from a trusted producer; digest refers to the original upload."""

    pixels: bytes
    width: _PositiveInt
    height: _PositiveInt
    image_digest: Sha256

    def _validate(self) -> None:
        _require(len(self.pixels) == self.width * self.height * 3, "RGB byte length does not match dimensions")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRef(_WireRecord):
    model_id: str
    model_revision: _CommitSha
    processor_revision: _CommitSha


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovedProfile(_Record):
    """Configuration supplied by a trusted caller, not proof of approval by itself.

    Prompt token audits require a positive integer in limits["prompt_token_limit"].
    """

    profile_id: str
    model: ModelRef
    config_sha256: Sha256
    device: str
    dtype: str
    limits: Mapping[str, _NonNegativeNumber]
    approval_ref: str
    tokenizer_revision: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalRef(_WireRecord):
    experiment_id: str
    run_id: str

    def _validate(self) -> None:
        match = _RUN_ID.fullmatch(self.run_id)
        _require(self.experiment_id in _EXPERIMENTS and match is not None and match[1] == self.experiment_id,
                 "run_id must match the experiment, phase, UTC timestamp and UUIDv4 format")
        try:
            datetime.strptime(match[3], "%Y%m%dT%H%M%SZ")
        except ValueError as exc:
            raise ContractError("run_id has an invalid UTC timestamp") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorInfo(_WireRecord):
    code: ErrorCode
    message_vi: str
    retryable: bool


class CNNStatus(str, Enum):
    ACCEPT = "ACCEPT"
    E7_REJECT = "E7_REJECT"
    OOD_REJECT = "OOD_REJECT"


class PolicyMode(str, Enum):
    BREED_CONTEXT_ALLOWED = "breed_context_allowed"
    OBSERVATION_ONLY = "observation_only"


class BranchOutcome(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    RESOURCE_EXHAUSTED = "resource_exhausted"


@dataclass(frozen=True, slots=True, kw_only=True)
class AllowedCNNContext(_WireRecord):
    status: CNNStatus
    class_id: ClassId | _Missing = MISSING
    canonical_name: str | _Missing = MISSING

    def _validate(self) -> None:
        _require((self.class_id is MISSING) == (self.canonical_name is MISSING),
                 "CNN class_id and canonical_name must appear together")
        if self.class_id is not MISSING:
            _require(self.status is CNNStatus.ACCEPT, "only ACCEPT can carry breed context",
                     ErrorCode.POLICY_BLOCKED)


def _bind_cnn_context(
    context: AllowedCNNContext | None,
    policy_mode: PolicyMode | None,
    names: Mapping[ClassId, str] | None,
) -> None:
    if context is not None and context.class_id is not MISSING:
        _require(policy_mode is PolicyMode.BREED_CONTEXT_ALLOWED, "breed context requires caller policy",
                 ErrorCode.POLICY_BLOCKED)
        _require(names is not None and names[context.class_id] == context.canonical_name,
                 "CNN context name does not match canonical mapping", ErrorCode.MAPPING_MISMATCH)


# Contract 01: knowledge and prompt shapes, plus pure byte/snapshot validation.


class SourceReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PromptType(str, Enum):
    NAME = "name"
    MORPHOLOGY = "morphology"


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceRecord(_WireRecord):
    source_id: str
    title: str
    url: str
    review_status: SourceReviewStatus
    review_ref: str | _Missing = MISSING

    def _validate(self) -> None:
        try:
            url = urlsplit(self.url)
        except ValueError as exc:
            raise ContractError("invalid source URL") from exc
        _require(url.scheme in {"http", "https"} and bool(url.hostname) and not any(c.isspace() for c in self.url),
                 "source URL must be HTTP(S)")


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphAttribute(_WireRecord):
    attribute_id: str
    value: str
    source_ids: tuple[str, ...]
    variability_note: str | _Missing = MISSING
    uncertainty_note: str | _Missing = MISSING

    def _validate(self) -> None:
        _require(bool(self.source_ids), "morphological attribute needs a source reference")
        _unique(self.source_ids, "attribute source_ids")


def _normalized_alias(alias: str) -> str:
    """Normalize aliases only for snapshot validation, not for lookup."""
    folded = unicodedata.normalize("NFC", alias).casefold()
    return " ".join(unicodedata.normalize("NFC", folded).split())


@dataclass(frozen=True, slots=True, kw_only=True)
class BreedRecord(_WireRecord):
    class_id: ClassId
    canonical_name: str
    aliases: Mapping[str, tuple[str, ...]]
    attributes: tuple[MorphAttribute, ...]

    def _validate(self) -> None:
        _require(self.aliases.keys() == {"vi", "en"}, "aliases must have vi and en lists")
        for values in self.aliases.values():
            _require(all(alias == alias.strip() for alias in values), "aliases must be trimmed")
            _unique(tuple(map(_normalized_alias, values)), "normalized aliases in one language")
        _unique(tuple(attribute.attribute_id for attribute in self.attributes), "breed attribute_ids")


@dataclass(frozen=True, slots=True, kw_only=True)
class BreedKnowledgeBase(_WireRecord):
    kb_version: str
    mapping_sha256: Sha256
    breeds: tuple[BreedRecord, ...]
    sources: tuple[SourceRecord, ...]
    schema_version: _SchemaVersion = SCHEMA_VERSION

    def _validate(self) -> None:
        _ordered_classes(self.breeds, "breeds")
        source_ids = tuple(source.source_id for source in self.sources)
        _unique(source_ids, "sources")
        known_sources = set(source_ids)
        for breed in self.breeds:
            for attribute in breed.attributes:
                _require(known_sources.issuperset(attribute.source_ids),
                         "attribute source_id is missing from the KB", ErrorCode.INVALID_SNAPSHOT)


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptEntry(_WireRecord):
    class_id: ClassId
    canonical_name: str
    prompt_text: str
    attribute_ids: tuple[str, ...] | _Missing = MISSING
    source_ids: tuple[str, ...] | _Missing = MISSING


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptBank(_WireRecord):
    bank_version: str
    prompt_type: PromptType
    mapping_sha256: Sha256
    entries: tuple[PromptEntry, ...]
    kb_version: str | _Missing = MISSING
    kb_sha256: Sha256 | _Missing = MISSING
    schema_version: _SchemaVersion = SCHEMA_VERSION

    def _validate(self) -> None:
        _ordered_classes(self.entries, "prompt entries")
        if self.prompt_type is PromptType.NAME:
            _require(self.kb_version is MISSING and self.kb_sha256 is MISSING, "name bank must omit KB fields")
            for entry in self.entries:
                _require(entry.attribute_ids is MISSING and entry.source_ids is MISSING,
                         "name entries must omit morphology provenance")
            return
        _require(self.kb_version is not MISSING and self.kb_sha256 is not MISSING,
                 "morphology bank requires KB version and hash")
        for entry in self.entries:
            _require(entry.attribute_ids is not MISSING and bool(entry.attribute_ids)
                     and entry.source_ids is not MISSING and bool(entry.source_ids),
                     "morphology entries require attribute and source IDs")
            _unique(entry.attribute_ids, "prompt attribute_ids")
            _unique(entry.source_ids, "prompt source_ids")
            _require(entry.canonical_name in entry.prompt_text, "morphology prompt must retain the canonical breed name")


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenAuditEntry(_WireRecord):
    class_id: ClassId
    token_count: _PositiveInt
    truncated: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenAudit(_WireRecord):
    prompt_bank_sha256: Sha256
    profile_id: str
    tokenizer_revision: str
    token_limit: _PositiveInt
    entries: tuple[TokenAuditEntry, ...]

    def _validate(self) -> None:
        _require(len(self.entries) == len(CLASS_IDS)
                 and {entry.class_id for entry in self.entries} == set(CLASS_IDS),
                 "token audit must cover every class exactly once")


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptTokenAudit(_WireRecord):
    audits: tuple[TokenAudit, ...]
    schema_version: _SchemaVersion = SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeSnapshot(_Record):
    data: BreedKnowledgeBase
    kb_sha256: Sha256


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSnapshot(_Record):
    data: PromptBank
    prompt_bank_sha256: Sha256


def _bind_mapping(records: tuple, digest: Sha256, mapping: CanonicalMapping) -> None:
    _require(digest == mapping.mapping_sha256, "canonical mapping SHA-256 mismatch", ErrorCode.MAPPING_MISMATCH)
    for record in records:
        _require(record.canonical_name == mapping.names[record.class_id], "canonical breed name mismatch",
                 ErrorCode.MAPPING_MISMATCH)


def validate_knowledge_snapshot(
    raw: bytes,
    *,
    mapping: CanonicalMapping,
    expected_sha256: Sha256,
    reviewed_sources: Mapping[str, str],
    allowed_attribute_ids: Collection[str],
) -> KnowledgeSnapshot:
    """Validate provided bytes against pins and review evidence chosen by the caller."""
    _expect_record(mapping, CanonicalMapping)
    verify_sha256(raw, expected_sha256)
    data = decode_record(raw, BreedKnowledgeBase)
    _bind_mapping(data.breeds, data.mapping_sha256, mapping)
    sources = {source.source_id: source for source in data.sources}
    for breed in data.breeds:
        for attribute in breed.attributes:
            _require(attribute.attribute_id in allowed_attribute_ids, "KB attribute ID is outside the supplied vocabulary",
                     ErrorCode.INVALID_SNAPSHOT)
            for source_id in attribute.source_ids:
                source = sources[source_id]
                _require(source.review_status is SourceReviewStatus.APPROVED
                         and source.review_ref is not MISSING
                         and reviewed_sources.get(source_id) == source.review_ref,
                         "KB source lacks matching trusted review evidence", ErrorCode.UNREVIEWED_SOURCE)
    return KnowledgeSnapshot(data=data, kb_sha256=expected_sha256)


def validate_prompt_snapshot(
    raw: bytes,
    *,
    mapping: CanonicalMapping,
    expected_sha256: Sha256,
    knowledge: KnowledgeSnapshot | None,
) -> PromptSnapshot:
    """Validate prompt bytes and, for morphology banks, bind breed-level provenance."""
    _expect_record(mapping, CanonicalMapping)
    verify_sha256(raw, expected_sha256)
    data = decode_record(raw, PromptBank)
    _bind_mapping(data.entries, data.mapping_sha256, mapping)
    if data.prompt_type is PromptType.MORPHOLOGY:
        _require(knowledge is not None, "morphology bank needs the trusted KB snapshot", ErrorCode.INVALID_SNAPSHOT)
        _expect_record(knowledge, KnowledgeSnapshot)
        _require(data.kb_version == knowledge.data.kb_version and data.kb_sha256 == knowledge.kb_sha256,
                 "morphology bank KB binding mismatch", ErrorCode.HASH_MISMATCH)
        _require(data.mapping_sha256 == knowledge.data.mapping_sha256,
                 "morphology bank and KB use different mappings", ErrorCode.MAPPING_MISMATCH)
        for entry in data.entries:
            attributes = {attribute.attribute_id: attribute
                          for attribute in knowledge.data.breeds[entry.class_id].attributes}
            _require(attributes.keys() >= set(entry.attribute_ids), "prompt cites unknown attributes for its breed",
                     ErrorCode.INVALID_SNAPSHOT)
            allowed_sources = {source_id for key in entry.attribute_ids for source_id in attributes[key].source_ids}
            _require(allowed_sources.issuperset(entry.source_ids),
                     "prompt sources must reference the listed attributes of the same breed", ErrorCode.INVALID_SNAPSHOT)
    return PromptSnapshot(data=data, prompt_bank_sha256=expected_sha256)


def validate_prompt_token_audit(
    audit: PromptTokenAudit, *, bank: PromptSnapshot, profile: ApprovedProfile,
) -> None:
    _expect_record(audit, PromptTokenAudit)
    _expect_record(bank, PromptSnapshot)
    _expect_record(profile, ApprovedProfile)
    trusted_limit = profile.limits.get("prompt_token_limit")
    _require(type(trusted_limit) is int and trusted_limit > 0,
             "trusted profile prompt_token_limit must be a positive integer", ErrorCode.INVALID_SNAPSHOT)
    matches = tuple(
        item for item in audit.audits
        if item.prompt_bank_sha256 == bank.prompt_bank_sha256
        and item.profile_id == profile.profile_id
        and item.tokenizer_revision == profile.tokenizer_revision
    )
    _require(len(matches) == 1, "expected one audit for the supplied bank, profile and tokenizer",
             ErrorCode.INVALID_SNAPSHOT)
    match = matches[0]
    _require(match.token_limit == trusted_limit,
             "audit token_limit does not match trusted profile prompt_token_limit", ErrorCode.INVALID_SNAPSHOT)
    _require(all(not entry.truncated and entry.token_count <= trusted_limit for entry in match.entries),
             "prompt tokenization exceeded its limit or was truncated", ErrorCode.TOKEN_TRUNCATION)


# Contract 02: describe and validate retrieval, without resolving mentions.


class RetrievalStatus(str, Enum):
    HIT = "hit"
    PARTIAL_HIT = "partial_hit"
    MISS = "miss"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"
    RETRIEVAL_ERROR = "retrieval_error"


class BreedLookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBudget(_WireRecord):
    max_tokens: _NonNegativeInt
    tokenizer_revision: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalRequest(_WireRecord):
    question: str
    explicit_class_ids: tuple[ClassId, ...]
    cnn_context: AllowedCNNContext | None
    kb_version: str
    kb_sha256: Sha256
    context_budget: ContextBudget
    lookup_applicable: bool
    schema_version: _SchemaVersion = SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class BreedLookup(_WireRecord):
    mention: str | None
    class_id: ClassId | None
    status: BreedLookupStatus
    candidate_class_ids: tuple[ClassId, ...]
    reason: str | _Missing = MISSING

    def _validate(self) -> None:
        if self.status is BreedLookupStatus.AMBIGUOUS:
            _require(self.class_id is None and len(self.candidate_class_ids) >= 2,
                     "ambiguous lookup requires multiple candidates and no resolved class")
            _require(self.candidate_class_ids == tuple(sorted(set(self.candidate_class_ids))),
                     "ambiguous candidate IDs must be unique and ascending")
            return
        _require(not self.candidate_class_ids, "only ambiguous lookups carry candidate IDs")
        if self.status is BreedLookupStatus.HIT:
            _require(self.class_id is not None, "a hit requires a resolved class")
        elif self.class_id is not None or self.reason == "budget_exhausted":
            _require(self.class_id is not None and self.reason == "budget_exhausted",
                     "a resolved miss must retain the class ID and report budget_exhausted")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedEntry(_WireRecord):
    class_id: ClassId
    canonical_name: str
    attributes: tuple[MorphAttribute, ...]

    def _validate(self) -> None:
        _require(bool(self.attributes), "retrieved entry needs at least one attribute")
        _unique(tuple(attribute.attribute_id for attribute in self.attributes), "retrieved attribute IDs")


@dataclass(frozen=True, slots=True, kw_only=True)
class TruncationInfo(_WireRecord):
    applied: bool
    budget_tokens: _NonNegativeInt
    used_tokens: _NonNegativeInt
    omitted_attribute_count: _NonNegativeInt

    def _validate(self) -> None:
        _require(self.used_tokens <= self.budget_tokens, "retrieval token budget exceeded", ErrorCode.BUDGET_EXCEEDED)
        _require(self.applied == (self.omitted_attribute_count > 0), "truncation flag and omitted count disagree")


_HIT_STATUSES = frozenset({RetrievalStatus.HIT, RetrievalStatus.PARTIAL_HIT})


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalResult(_WireRecord):
    status: RetrievalStatus
    per_breed: tuple[BreedLookup, ...]
    resolved_class_ids: tuple[ClassId, ...]
    entries: tuple[RetrievedEntry, ...]
    context: _Context
    source_ids: tuple[str, ...]
    kb_version: str
    kb_sha256: Sha256
    truncation: TruncationInfo
    latency_ms: _NonNegativeNumber
    error: ErrorInfo | _Missing = MISSING
    schema_version: _SchemaVersion = SCHEMA_VERSION

    def _validate(self) -> None:
        _require((self.error is not MISSING) == (self.status is RetrievalStatus.RETRIEVAL_ERROR),
                 "error metadata is required only for retrieval_error")
        resolved = tuple(item.class_id for item in self.per_breed if item.class_id is not None)
        _unique(resolved, "resolved class IDs")
        _require(self.resolved_class_ids == resolved, "resolved_class_ids must match per_breed")
        hits = tuple(item.class_id for item in self.per_breed if item.status is BreedLookupStatus.HIT)
        _require(hits == tuple(entry.class_id for entry in self.entries), "entries must match ordered breed hits")
        expected_sources = {s for entry in self.entries for attribute in entry.attributes for s in attribute.source_ids}
        _unique(self.source_ids, "retrieved source_ids")
        _require(set(self.source_ids) == expected_sources, "source_ids must match retrieved attributes")

        if self.status in (RetrievalStatus.NOT_APPLICABLE, RetrievalStatus.RETRIEVAL_ERROR):
            _require(not self.per_breed, "not_applicable and retrieval_error cannot carry breed lookups")
        else:
            if hits:
                expected = RetrievalStatus.HIT if len(hits) == len(self.per_breed) else RetrievalStatus.PARTIAL_HIT
            elif any(item.status is BreedLookupStatus.AMBIGUOUS for item in self.per_breed):
                expected = RetrievalStatus.AMBIGUOUS
            else:
                expected = RetrievalStatus.MISS
            _require(self.status is expected, "overall retrieval status disagrees with per_breed")
        _require(bool(self.context) == (self.status in _HIT_STATUSES), "context is present only for retrieval hits")
        if not self.context:
            _require(self.truncation.used_tokens == 0, "empty context must use zero KB tokens")


class ContextTokenCounter(Protocol):
    def __call__(self, context: str) -> int: ...


class KnowledgeRetriever(Protocol):
    def __call__(self, request: RetrievalRequest, *, knowledge: KnowledgeSnapshot,
                 count_context_tokens: ContextTokenCounter) -> RetrievalResult: ...


def count_context_tokens(counter: ContextTokenCounter, context: str) -> int:
    """Check a caller-provided token count without implementing a tokenizer."""
    _field_value(context, _Context, "context", wire=False)
    count = counter(context)
    _require(type(count) is int and count >= 0, "context token count must be a nonnegative integer")
    return count


def _bind_knowledge(version: str, digest: Sha256, knowledge: KnowledgeSnapshot) -> None:
    _require(version == knowledge.data.kb_version and digest == knowledge.kb_sha256,
             "KB version or hash mismatch", ErrorCode.HASH_MISMATCH)


def validate_retrieval_request(
    request: RetrievalRequest, *, knowledge: KnowledgeSnapshot, policy_mode: PolicyMode | None = None,
) -> None:
    """Bind a request to a trusted pinned KB and caller-supplied policy."""
    _expect_record(request, RetrievalRequest)
    _expect_record(knowledge, KnowledgeSnapshot)
    _require(policy_mode is None or type(policy_mode) is PolicyMode, "invalid trusted retrieval policy")
    _bind_knowledge(request.kb_version, request.kb_sha256, knowledge)
    if request.lookup_applicable:
        _require(policy_mode is PolicyMode.BREED_CONTEXT_ALLOWED, "lookup requires caller policy",
                 ErrorCode.POLICY_BLOCKED)
    _bind_cnn_context(request.cnn_context, policy_mode,
                      {breed.class_id: breed.canonical_name for breed in knowledge.data.breeds})


@dataclass(frozen=True, slots=True, init=False)
class ValidatedRetrieval:
    """In-process retrieval result bound to a trusted snapshot and renderer."""

    result: RetrievalResult
    request: RetrievalRequest
    policy_mode: PolicyMode | None

    def __init__(self) -> None:
        raise TypeError("use validate_retrieval_result()")


def validate_retrieval_result(
    result: RetrievalResult,
    *,
    request: RetrievalRequest,
    knowledge: KnowledgeSnapshot,
    token_counter: ContextTokenCounter,
    render_context: Callable[[tuple[RetrievedEntry, ...]], str],
    policy_mode: PolicyMode | None = None,
) -> ValidatedRetrieval:
    """Bind a retrieval result before passing it to an assistant.

    The caller supplies a trusted renderer independent of result.context.
    It receives only the ordered entries already checked against the KB.
    The binding verifies exact context text. It does not evaluate its claims.
    """
    _expect_record(result, RetrievalResult)
    validate_retrieval_request(request, knowledge=knowledge, policy_mode=policy_mode)
    _bind_knowledge(result.kb_version, result.kb_sha256, knowledge)
    _require((result.status is RetrievalStatus.NOT_APPLICABLE) == (not request.lookup_applicable),
             "retrieval result does not match lookup applicability", ErrorCode.POLICY_BLOCKED)
    if request.lookup_applicable and result.status is not RetrievalStatus.RETRIEVAL_ERROR:
        explicit = tuple(dict.fromkeys(request.explicit_class_ids))
        _require(result.resolved_class_ids[:len(explicit)] == explicit,
                 "resolved class IDs must start with the ordered explicit IDs")
        cnn = request.cnn_context
        if cnn is not None and cnn.class_id is not MISSING and cnn.class_id not in explicit:
            uses_cnn_fallback = any(
                item.mention is None and item.class_id == cnn.class_id for item in result.per_breed
            )
            _require(not uses_cnn_fallback or all(item.mention is None for item in result.per_breed),
                     "CNN fallback may not replace or supplement an explicit breed mention", ErrorCode.POLICY_BLOCKED)
    _require(result.truncation.budget_tokens == request.context_budget.max_tokens,
             "retrieval result budget does not match request", ErrorCode.BUDGET_EXCEEDED)
    for entry in result.entries:
        breed = knowledge.data.breeds[entry.class_id]
        _require(entry.canonical_name == breed.canonical_name, "retrieved breed name mismatch",
                 ErrorCode.MAPPING_MISMATCH)
        selected = {attribute.attribute_id for attribute in entry.attributes}
        expected = tuple(attribute for attribute in breed.attributes if attribute.attribute_id in selected)
        _require(entry.attributes == expected, "retrieved attributes do not match pinned KB", ErrorCode.INVALID_SNAPSHOT)
    expected_context = render_context(result.entries) if result.entries else ""
    _require(type(expected_context) is str and result.context == expected_context,
             "retrieval context does not match trusted rendered entries", ErrorCode.INVALID_OUTPUT)
    if result.context:
        _require(result.truncation.used_tokens == count_context_tokens(token_counter, result.context),
                 "retrieval token count does not match the context", ErrorCode.BUDGET_EXCEEDED)
    validated = object.__new__(ValidatedRetrieval)
    object.__setattr__(validated, "result", result)
    object.__setattr__(validated, "request", request)
    object.__setattr__(validated, "policy_mode", policy_mode)
    return validated


# Contracts 03 and 04: shared image operations and adapter results.


@dataclass(frozen=True, slots=True, kw_only=True)
class _ImageRequest(_Record):
    image: ValidatedRgb
    image_digest: Sha256
    profile_id: str
    eval_ref: EvalRef | _Missing = MISSING
    schema_version: _SchemaVersion = SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class _AdapterResult(_WireRecord):
    status: BranchOutcome
    image_digest: Sha256
    profile_id: str
    model: ModelRef
    latency_ms: _NonNegativeNumber | None
    eval_ref: EvalRef | _Missing = MISSING
    error: ErrorInfo | _Missing = MISSING
    schema_version: _SchemaVersion = SCHEMA_VERSION


_ZERO_SHOT_EXPERIMENTS = frozenset({"E12-A", "E12-B", "E13-A", "E13-B"})
_ASSISTANT_EXPERIMENTS = frozenset({"E14-A", "E14-B"})


def _check_eval_experiment(eval_ref: EvalRef | _Missing, experiments: frozenset[str]) -> None:
    _require(eval_ref is MISSING or eval_ref.experiment_id in experiments,
             "eval_ref does not belong to this adapter")


def _validate_image_request(request: _ImageRequest, experiments: frozenset[str]) -> None:
    _require(request.image_digest == request.image.image_digest, "request digest does not match validated RGB metadata",
             ErrorCode.HASH_MISMATCH)
    _check_eval_experiment(request.eval_ref, experiments)


def _validate_outcome(result: _AdapterResult, experiments: frozenset[str]) -> None:
    _check_eval_experiment(result.eval_ref, experiments)
    _require((result.error is not MISSING) == (result.status is not BranchOutcome.OK),
             "only failed adapter results require error metadata", ErrorCode.INVALID_OUTPUT)


def _bind_image_request(request: _ImageRequest, profile: ApprovedProfile, offline: bool) -> None:
    _expect_record(profile, ApprovedProfile)
    _require(type(offline) is bool, "offline must be chosen by the caller")
    _require((request.eval_ref is not MISSING) == offline, "eval_ref is required only for offline requests")
    _require(request.profile_id == profile.profile_id, "request profile does not match trusted profile")


def _bind_adapter_result(result: _AdapterResult, request: _ImageRequest, profile: ApprovedProfile) -> None:
    _require(result.eval_ref == request.eval_ref, "adapter result eval_ref mismatch", ErrorCode.INVALID_OUTPUT)
    _require(result.image_digest == request.image_digest, "adapter result image digest mismatch", ErrorCode.HASH_MISMATCH)
    _require(result.profile_id == profile.profile_id and result.model == profile.model,
             "adapter result model or profile mismatch", ErrorCode.INVALID_OUTPUT)


class ScoreType(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    SCALED_SIMILARITY = "scaled_similarity"
    SIMILARITY = "similarity"
    LOGIT = "logit"


def rank_scores(scores: tuple[_Number, ...] | list[_Number]) -> tuple[ClassId, ...]:
    """Return all canonical IDs in score-descending, ID-ascending order."""
    values = _field_value(scores, tuple[_Number, ...], "scores", wire=False)
    _require(len(values) == len(CLASS_IDS), "ranking requires exactly 30 scores", ErrorCode.INVALID_OUTPUT)
    return tuple(sorted(CLASS_IDS, key=lambda class_id: (-values[class_id], class_id)))


@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroShotRequest(_ImageRequest):
    prompt_bank_sha256: Sha256
    mapping_sha256: Sha256

    def _validate(self) -> None:
        _validate_image_request(self, _ZERO_SHOT_EXPERIMENTS)


@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroShotResult(_AdapterResult):
    mapping_sha256: Sha256
    prompt_bank_sha256: Sha256
    calibrated: Literal[False]
    score_type: ScoreType | _Missing = MISSING
    scores: tuple[_Number, ...] | _Missing = MISSING
    top1_class_id: ClassId | _Missing = MISSING
    top5_class_ids: tuple[ClassId, ...] | _Missing = MISSING

    def _validate(self) -> None:
        _validate_outcome(self, _ZERO_SHOT_EXPERIMENTS)
        scoring = (self.score_type, self.scores, self.top1_class_id, self.top5_class_ids)
        if self.status is not BranchOutcome.OK:
            _require(all(item is MISSING for item in scoring), "failed classification must omit ranking",
                     ErrorCode.INVALID_OUTPUT)
            return
        _require(all(item is not MISSING for item in scoring), "ok classification requires scores and ranking",
                 ErrorCode.INVALID_OUTPUT)
        expected = rank_scores(self.scores)
        _require(self.top1_class_id == expected[0] and self.top5_class_ids == expected[:5],
                 "Top-1 and Top-5 do not match deterministic ranking", ErrorCode.INVALID_OUTPUT)


class ZeroShotClassifier(Protocol):
    def classify(self, request: ZeroShotRequest, *, profile: ApprovedProfile,
                 prompts: PromptSnapshot) -> ZeroShotResult: ...


def validate_zero_shot_request(
    request: ZeroShotRequest, *, profile: ApprovedProfile, prompts: PromptSnapshot, offline: bool = False,
) -> None:
    _expect_record(request, ZeroShotRequest)
    _expect_record(prompts, PromptSnapshot)
    _bind_image_request(request, profile, offline)
    _require(request.mapping_sha256 == prompts.data.mapping_sha256,
             "request mapping does not match trusted prompt snapshot", ErrorCode.MAPPING_MISMATCH)
    _require(request.prompt_bank_sha256 == prompts.prompt_bank_sha256,
             "request prompt hash does not match trusted snapshot", ErrorCode.HASH_MISMATCH)
    if offline:
        morphology = request.eval_ref.experiment_id.endswith("-B")
        _require((prompts.data.prompt_type is PromptType.MORPHOLOGY) == morphology,
                 "offline experiment does not match prompt bank type")


def validate_zero_shot_result(
    result: ZeroShotResult, *, request: ZeroShotRequest, profile: ApprovedProfile,
    prompts: PromptSnapshot, offline: bool = False,
) -> None:
    _expect_record(result, ZeroShotResult)
    validate_zero_shot_request(request, profile=profile, prompts=prompts, offline=offline)
    _bind_adapter_result(result, request, profile)
    _require(result.mapping_sha256 == request.mapping_sha256, "classifier result mapping mismatch",
             ErrorCode.MAPPING_MISMATCH)
    _require(result.prompt_bank_sha256 == request.prompt_bank_sha256, "classifier result prompt hash mismatch",
             ErrorCode.HASH_MISMATCH)


class Operation(str, Enum):
    ATTRIBUTES = "attributes"
    VQA = "vqa"
    ATTRIBUTES_AND_VQA = "attributes_and_vqa"


class Visibility(str, Enum):
    OBSERVED = "observed"
    UNKNOWN = "unknown"
    NOT_VISIBLE = "not_visible"
    NOT_APPLICABLE = "not_applicable"


class Disposition(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    NOT_APPLICABLE = "not_applicable"


class Uncertainty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNSPECIFIED = "unspecified"


ATTRIBUTE_VOCABULARY_V1 = MappingProxyType({
    "ear_position": ("erect", "semi_erect", "dropped"),
    "ear_shape": ("triangular", "rounded", "elongated"),
    "coat_color": ("black", "white", "gray", "brown", "tan", "cream", "red", "golden"),
    "coat_length": ("short", "medium", "long"),
    "coat_texture": ("straight", "wavy", "curly", "wiry"),
    "muzzle_length": ("short", "medium", "long"),
    "tail_shape": ("straight", "curved", "curled"),
    "body_build": ("slender", "medium", "stocky"),
})


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributeObservation(_WireRecord):
    attribute_id: str
    visibility: Visibility
    value: str | tuple[str, ...] | None

    def _validate(self) -> None:
        _require(self.attribute_id in ATTRIBUTE_VOCABULARY_V1, "unknown attribute_id")
        if self.visibility is not Visibility.OBSERVED:
            _require(self.value is None, "unobserved attributes require a null value")
            return
        options = ATTRIBUTE_VOCABULARY_V1[self.attribute_id]
        if self.attribute_id == "coat_color":
            _require(isinstance(self.value, tuple) and bool(self.value), "observed coat_color requires a nonempty array")
            _require(self.value == tuple(color for color in options if color in self.value),
                     "coat_color must use unique vocabulary values in palette order")
        else:
            _require(isinstance(self.value, str) and self.value in options,
                     "observed attribute requires a single controlled value")


def _unique_observations(observations: tuple[AttributeObservation, ...]) -> None:
    _unique(tuple(item.attribute_id for item in observations), "observation attribute_ids")


def _context_digest(context: str) -> Sha256:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


_EMPTY_CONTEXT_HASH = _context_digest("")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextProvenance(_WireRecord):
    retrieval_status: RetrievalStatus
    kb_version: str | None
    kb_sha256: Sha256 | None
    context_sha256: Sha256

    def _validate(self) -> None:
        _require((self.kb_version is None) == (self.kb_sha256 is None), "KB version and hash must appear together")
        if self.kb_version is None:
            _require(self.retrieval_status is RetrievalStatus.NOT_APPLICABLE, "no-KB retrieval must be not_applicable")
        _require((self.context_sha256 == _EMPTY_CONTEXT_HASH) == (self.retrieval_status not in _HIT_STATUSES),
                 "KB context hash must agree with retrieval hit status", ErrorCode.HASH_MISMATCH)


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantRequest(_ImageRequest):
    operation: Operation
    question: str | None
    cnn_context: AllowedCNNContext | None
    intermediate_attributes: tuple[AttributeObservation, ...] | None
    kb_context: _Context
    allowed_source_ids: tuple[str, ...]
    context_provenance: ContextProvenance
    policy_mode: PolicyMode

    def _validate(self) -> None:
        _validate_image_request(self, _ASSISTANT_EXPERIMENTS)
        _require((self.question is None) == (self.operation is Operation.ATTRIBUTES),
                 "question must be null for attributes and present for VQA")
        if self.intermediate_attributes is not None:
            _unique_observations(self.intermediate_attributes)
        _unique(self.allowed_source_ids, "allowed_source_ids")
        provenance = self.context_provenance
        has_hit = provenance.retrieval_status in _HIT_STATUSES
        _require(has_hit == bool(self.kb_context) == bool(self.allowed_source_ids),
                 "only retrieval hits can carry KB context and sources")
        _require(provenance.context_sha256 == _context_digest(self.kb_context),
                 "KB context SHA-256 mismatch", ErrorCode.HASH_MISMATCH)
        if self.eval_ref is not MISSING:
            if self.eval_ref.experiment_id == "E14-A":
                _require(provenance.kb_version is None, "E14-A must not carry KB provenance")
            elif self.eval_ref.experiment_id == "E14-B" and self.operation is not Operation.ATTRIBUTES:
                _require(provenance.kb_version is not None, "E14-B VQA requires a KB version and hash",
                         ErrorCode.INVALID_SNAPSHOT)
        if self.policy_mode is PolicyMode.OBSERVATION_ONLY:
            _require(not self.kb_context and (self.cnn_context is None or self.cnn_context.class_id is MISSING),
                     "observation_only must not carry breed context", ErrorCode.POLICY_BLOCKED)


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratedContent(_WireRecord):
    attributes: tuple[AttributeObservation, ...]
    answer_vi: str | None
    disposition: Disposition
    uncertainty: Uncertainty
    abstention_reason: str | None
    source_ids: tuple[str, ...]

    def _validate(self) -> None:
        _unique_observations(self.attributes)
        _unique(self.source_ids, "generated source_ids")
        if self.disposition is Disposition.ABSTAINED:
            _require(self.abstention_reason is not None, "abstained output needs a reason", ErrorCode.INVALID_OUTPUT)
        else:
            _require(self.abstention_reason is None, "only abstained output can carry a reason", ErrorCode.INVALID_OUTPUT)
            _require((self.answer_vi is not None) == (self.disposition is Disposition.ANSWERED),
                     "answer does not match disposition", ErrorCode.INVALID_OUTPUT)


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantResult(_AdapterResult):
    context_provenance: ContextProvenance
    generated: GeneratedContent | _Missing = MISSING

    def _validate(self) -> None:
        _validate_outcome(self, _ASSISTANT_EXPERIMENTS)
        _require((self.generated is not MISSING) == (self.status is BranchOutcome.OK),
                 "generated content is required only for ok assistant results", ErrorCode.INVALID_OUTPUT)
        if self.eval_ref is not MISSING and self.eval_ref.experiment_id == "E14-A":
            _require(self.context_provenance.kb_version is None, "E14-A must not carry KB provenance")


class GenerativeAssistant(Protocol):
    def extract_attributes(self, request: AssistantRequest, *, profile: ApprovedProfile) -> AssistantResult: ...

    def answer_question(self, request: AssistantRequest, *, profile: ApprovedProfile) -> AssistantResult: ...

    def execute(self, request: AssistantRequest, *, profile: ApprovedProfile) -> AssistantResult: ...


def validate_assistant_request(
    request: AssistantRequest,
    *,
    profile: ApprovedProfile,
    policy_mode: PolicyMode,
    mapping: CanonicalMapping | None = None,
    retrieval: ValidatedRetrieval | None = None,
    offline: bool = False,
) -> None:
    _expect_record(request, AssistantRequest)
    if mapping is not None:
        _expect_record(mapping, CanonicalMapping)
    _bind_image_request(request, profile, offline)
    _require(request.policy_mode is policy_mode, "request policy does not match trusted caller policy",
             ErrorCode.POLICY_BLOCKED)
    _bind_cnn_context(request.cnn_context, policy_mode, None if mapping is None else mapping.names)
    provenance = request.context_provenance
    if provenance.kb_version is None:
        _require(retrieval is None, "no-KB requests must omit retrieval binding")
        return
    _require(retrieval is not None, "KB provenance requires a trusted retrieval result", ErrorCode.INVALID_SNAPSHOT)
    _expect_record(retrieval, ValidatedRetrieval)
    lookup_request = retrieval.request
    if request.operation is not Operation.ATTRIBUTES:
        _require(request.question == lookup_request.question, "assistant and retrieval question mismatch",
                 ErrorCode.INVALID_SNAPSHOT)
    _require(request.cnn_context == lookup_request.cnn_context, "assistant and retrieval CNN context mismatch",
             ErrorCode.INVALID_SNAPSHOT)
    _require(policy_mode is retrieval.policy_mode, "assistant and retrieval policy mismatch",
             ErrorCode.POLICY_BLOCKED)
    _require(profile.tokenizer_revision == lookup_request.context_budget.tokenizer_revision,
             "assistant profile and retrieval tokenizer revision mismatch", ErrorCode.INVALID_SNAPSHOT)
    lookup = retrieval.result
    _require(provenance.kb_version == lookup.kb_version and provenance.kb_sha256 == lookup.kb_sha256,
             "assistant and retrieval KB pins disagree", ErrorCode.HASH_MISMATCH)
    _require(provenance.retrieval_status is lookup.status and request.kb_context == lookup.context
             and set(request.allowed_source_ids) == set(lookup.source_ids),
             "assistant context or sources do not match trusted retrieval", ErrorCode.INVALID_SNAPSHOT)


def decode_generated_content(raw: bytes, *, request: AssistantRequest) -> GeneratedContent:
    """Decode model-only fields; only the server can construct the result envelope."""
    content = decode_record(raw, GeneratedContent)
    validate_generated_content(content, request=request)
    return content


def validate_generated_content(content: GeneratedContent, *, request: AssistantRequest) -> None:
    _expect_record(content, GeneratedContent)
    _expect_record(request, AssistantRequest)
    _require((content.disposition is Disposition.NOT_APPLICABLE) == (request.operation is Operation.ATTRIBUTES),
             "generated disposition does not match the requested operation", ErrorCode.INVALID_OUTPUT)
    _require(set(request.allowed_source_ids).issuperset(content.source_ids),
             "generated content cites a source outside the request allowlist", ErrorCode.INVALID_OUTPUT)


def validate_assistant_result(
    result: AssistantResult,
    *,
    request: AssistantRequest,
    profile: ApprovedProfile,
    policy_mode: PolicyMode,
    mapping: CanonicalMapping | None = None,
    retrieval: ValidatedRetrieval | None = None,
    offline: bool = False,
) -> None:
    _expect_record(result, AssistantResult)
    validate_assistant_request(request, profile=profile, policy_mode=policy_mode, mapping=mapping,
                               retrieval=retrieval, offline=offline)
    _bind_adapter_result(result, request, profile)
    _require(result.context_provenance == request.context_provenance,
             "assistant result context binding mismatch", ErrorCode.INVALID_OUTPUT)
    if result.generated is not MISSING:
        validate_generated_content(result.generated, request=request)


# Contract 05: envelopes and snapshots, without session or publication behavior.


class CNNState(str, Enum):
    DECIDED = "decided"
    NOT_RUN = "not_run"
    ERROR = "error"


class CNNMode(str, Enum):
    STANDARD = "standard"
    FIT = "fit"


class Branch(str, Enum):
    COMPARISON = "comparison"
    ASSISTANT = "assistant"


class RuntimeStatus(str, Enum):
    DISABLED = "disabled"
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    RESOURCE_EXHAUSTED = "resource_exhausted"


@dataclass(frozen=True, slots=True, kw_only=True)
class CNNDecisionSnapshot(_WireRecord):
    state: CNNState
    status: CNNStatus | None
    class_id: ClassId | None

    def _validate(self) -> None:
        _require((self.status is not None) == (self.state is CNNState.DECIDED),
                 "only decided CNN snapshots have a status")
        _require((self.class_id is not None) == (self.status in (CNNStatus.ACCEPT, CNNStatus.E7_REJECT)),
                 "only ACCEPT or E7_REJECT CNN snapshots carry a class_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class _BranchIdentity(_Record):
    request_id: str
    session_id: str
    image_digest: Sha256
    revision: _NonNegativeInt
    cnn_mode: CNNMode
    branch: Branch
    profile_id: str
    schema_version: _SchemaVersion = SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class BranchRequest(_BranchIdentity, _WireRecord):
    cnn_decision: CNNDecisionSnapshot
    remaining_budget_ms: _NonNegativeNumber


def _bind_runtime_payload(envelope: _BranchIdentity, payload: _ImageRequest | _AdapterResult) -> None:
    _require(payload.eval_ref is MISSING, "runtime worker payload must omit eval_ref")
    _require(payload.image_digest == envelope.image_digest, "worker payload digest mismatch", ErrorCode.HASH_MISMATCH)
    _require(payload.profile_id == envelope.profile_id, "worker payload profile mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerJob(_BranchIdentity):
    """Python-only job until the IPC carrier for validated RGB has been approved."""

    remaining_budget_ms: _NonNegativeNumber
    payload: ZeroShotRequest | AssistantRequest

    def _validate(self) -> None:
        expected = ZeroShotRequest if self.branch is Branch.COMPARISON else AssistantRequest
        _require(type(self.payload) is expected, "worker job payload must match its branch")
        _bind_runtime_payload(self, self.payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerReply(_BranchIdentity, _WireRecord):
    result: ZeroShotResult | AssistantResult

    def _validate(self) -> None:
        expected = ZeroShotResult if self.branch is Branch.COMPARISON else AssistantResult
        _require(type(self.result) is expected, "worker reply result must match its branch")
        _bind_runtime_payload(self, self.result)


def validate_worker_identity(expected: BranchRequest | WorkerJob, received: WorkerJob | WorkerReply) -> None:
    """Check identity equality. Freshness and publication belong to E15."""
    _require(type(expected) in (BranchRequest, WorkerJob) and type(received) in (WorkerJob, WorkerReply),
             "worker identity requires validated request and worker envelopes")
    for name in ("request_id", "session_id", "image_digest", "revision", "cnn_mode", "branch", "profile_id"):
        _require(getattr(expected, name) == getattr(received, name), f"worker identity mismatch: {name}")


@dataclass(frozen=True, slots=True, kw_only=True)
class BranchState(_WireRecord):
    status: RuntimeStatus
    request_id: str | None
    error: ErrorInfo | _Missing = MISSING
    controller_response_vi: str | _Missing = MISSING
    model_result: ZeroShotResult | AssistantResult | _Missing = MISSING

    def _validate(self) -> None:
        if self.model_result is not MISSING:
            _require(self.status is RuntimeStatus.OK and self.model_result.status is BranchOutcome.OK,
                     "only ok branch states may contain a successful model result")
            _require(self.model_result.eval_ref is MISSING, "runtime branch state must omit eval_ref")


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionState(_Record):
    session_id: str
    revision: _NonNegativeInt
    image_digest: Sha256 | None
    cnn_mode: CNNMode
    cnn_decision: CNNDecisionSnapshot
    latest_request_ids: Mapping[Branch, str | None]
    selected_profile_ids: Mapping[Branch, str | None]
    branch_states: Mapping[Branch, BranchState]

    def _validate(self) -> None:
        _require(all(mapping.keys() == set(Branch) for mapping in
                     (self.latest_request_ids, self.selected_profile_ids, self.branch_states)),
                 "session maps must contain both branches")
        for branch, state in self.branch_states.items():
            if state.model_result is not MISSING:
                expected = ZeroShotResult if branch is Branch.COMPARISON else AssistantResult
                _require(type(state.model_result) is expected, "session model result does not match its branch")


__all__ = [
    "SCHEMA_VERSION", "CLASS_IDS", "MISSING", "ClassId", "Sha256", "CanonicalMapping", "ValidatedRgb",
    "ModelRef", "ApprovedProfile", "EvalRef", "ErrorInfo", "ErrorCode", "ContractError", "decode_json",
    "decode_record", "encode_record", "decode_canonical_mapping", "validate_class_id", "validate_sha256",
    "verify_sha256",
    "SourceReviewStatus", "PromptType", "SourceRecord", "MorphAttribute", "BreedRecord",
    "BreedKnowledgeBase", "PromptEntry", "PromptBank", "TokenAuditEntry", "TokenAudit", "PromptTokenAudit",
    "KnowledgeSnapshot", "PromptSnapshot", "validate_knowledge_snapshot", "validate_prompt_snapshot",
    "validate_prompt_token_audit",
    "RetrievalStatus", "BreedLookupStatus", "ContextBudget", "RetrievalRequest", "BreedLookup",
    "RetrievedEntry", "TruncationInfo", "RetrievalResult", "ValidatedRetrieval",
    "ContextTokenCounter", "KnowledgeRetriever",
    "count_context_tokens", "validate_retrieval_request", "validate_retrieval_result",
    "ScoreType", "ZeroShotRequest", "ZeroShotResult", "ZeroShotClassifier", "rank_scores",
    "validate_zero_shot_request", "validate_zero_shot_result",
    "Operation", "Visibility", "Disposition", "Uncertainty", "PolicyMode", "ATTRIBUTE_VOCABULARY_V1",
    "AttributeObservation", "AllowedCNNContext", "ContextProvenance", "AssistantRequest", "GeneratedContent",
    "AssistantResult", "GenerativeAssistant", "decode_generated_content", "validate_generated_content",
    "validate_assistant_request", "validate_assistant_result",
    "CNNState", "CNNStatus", "CNNMode", "Branch", "BranchOutcome", "RuntimeStatus",
    "CNNDecisionSnapshot", "BranchRequest", "WorkerJob", "WorkerReply", "BranchState", "SessionState",
    "validate_worker_identity",
]
