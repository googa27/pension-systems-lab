# ADR 0001: immutable snapshot portability

Status: accepted. Owner: publication-maintainers.

## Evidence and decision

Python 3.12's generated setter for a frozen/slotted dataclass raised `TypeError: super(type, obj)` when assigning the read-only `records` property or an unknown attribute. The same public API must instead consistently reject mutation with `FrozenInstanceError`; accepting arbitrary exception types would weaken its contract.

Use the maintained stdlib `dataclass(frozen=True, init=False)` for `IdentityRegistrySnapshot`, without generated slots/weakref slots. Ordinary dataclasses remain weak-referenceable. Per-record classes retain slots. Existing immutable mapping/bytes and identity seals are unchanged; lookup detects underlying-index replacement before returning a record.

The additional instance dictionary costs constant overhead per snapshot, not per row. Its direct-mutation route is covered by a new tampering regression. This is an API integrity mechanism, not a sandbox against arbitrary Python code. Profile snapshot overhead before considering another maintained immutable representation; no custom setter or monkeypatch is justified.

## Verification

The original Python 3.12 failure was reproduced before changing code. New property/field/unknown-attribute and dictionary-tampering tests were run RED, then all 45 identity/portability tests passed on Python 3.12. The full Python 3.12/3.13 matrix remains mandatory before merge.
