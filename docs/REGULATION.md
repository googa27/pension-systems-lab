# Chile regulation context (dated research note)

**Status date:** 2026-10-04. This note is documentation context only. It is not a hard-coded legal rule engine, legal advice, actuarial certification, or a guarantee that a calculation reflects current law.

## Official sources consulted

- BCN Ley Chile, Decreto Ley N° 3.500, *Establece Nuevo Sistema de Pensiones*: <https://www.bcn.cl/leychile/navegar?idNorma=7147>
- BCN Ley Chile, Ley N° 21.735, 2025 pension reform: <https://www.bcn.cl/leychile/navegar?idNorma=1212060>
- Superintendencia de Pensiones, Ley N° 21.735 normative page: <https://www.spensiones.cl/portal/institucional/594/w3-article-16483.html>

## Why this matters for the lab

- DL 3.500 is the base Chilean individual-capitalization pension-system statute.
- Ley 21.735, published 2025-03-26, creates a mixed pension system and social insurance in the contributory pillar, improves PGU, and modifies regulatory structures.
- BCN notes that Ley 21.735 has a general 2027-04-01 effectiveness frame with transitional exceptions.

## Modelling rule

The repository may document source context and dated assumptions, but it must not silently encode them as official current law. Any scenario using Chilean policy parameters must name its source, date, simplifications, and validation status.

## Implementation posture

- `pension-lab` demos are numerical/software smoke tests.
- Regulation-sensitive scenarios must be reviewed by a qualified domain professional before being described as legal, actuarial, or administrative outputs.
- Future source adapters must include source URL, retrieval date, checksum, unit, coverage, license/terms note, and test fixtures.
