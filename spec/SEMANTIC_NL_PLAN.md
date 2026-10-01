# Semantic multilingual NL plan profile

Profile: `semantic-v1`. Envelope: `wellmanifest.nl-plan/v1`. Status: candidate extension, 2026-10-01.

## Compatibility and authority

This optional profile replaces sentence-specific rules, synonym dictionaries, ASCII word extraction and mandatory lexical overlap **within semantic-v1**. Existing deterministic PL/EN behavior remains the legacy profile; adopters select a profile explicitly. Only exact, validated API/DSL syntax may bypass interpretation. Natural-language requests, including negation, never pass through the legacy matcher first.

The host supplies an operation catalog with stable `uri`, human-readable `desc`, `input_schema`, `output_schema`, explicit `effects`, declaration status and source `digest`. Only declared operations whose effects are allowed by the host enter retrieval. Missing or unknown effects grant no execution authority. Model output cannot register an operation, change policy, select credentials or overwrite an execution binding.

## Retrieval and translation

Preserve original Unicode, paths, quantities and identifiers. Embed the query and public contract descriptions; retrieve bounded candidates without a prior language-specific subject gate. Exact stable identifiers may augment retrieval, but must not fabricate arguments. Cache operation vectors by immutable embedding model identity and the complete public contract digest. Recompute changed contracts; query vectors need not be persisted. Resident models and bounded concurrency avoid per-request loading.

A small generator receives the original request, selected contracts and a bounded typed output schema. A compact provider wire format may use ordered calls and null non-applicable fields; the host constructs the canonical envelope, binds digests and validates each operation/argument pairing independently. Pass the schema to both its constrained decoder and its prompt: backend grammar support varies. Schema rejection, truncation or invalid JSON must not trigger an unconstrained code-generation fallback. Never infer success from a provider's successful HTTP response.

## Intermediate representation

The envelope has exactly `schema`, `status` and the status-specific payload. `ok` contains `plan`; `clarify` contains a nonempty `question`; `unsupported` contains a nonempty `reason`. Neither non-ok status contains executable steps.

A call has `kind: call`, `operation` (catalog URI), `arguments` (that operation's input schema), and `digest` (SHA-256 of its canonical public contract). A sequence has `kind: sequence` and 1–16 ordered literal calls. No result references, variables, conditions or loops are supported in v1. Requests requiring those constructs must be clarified or reported unsupported. Missing required arguments must never be invented. JSON validity alone does not establish correct direction, negation or intent.

The generic envelope schema is structural documentation; runtimes MUST specialize operation and digest to catalog constants and arguments to their corresponding input schema. Document-local references must retain their original meaning when schemas are relocated. External schema resolution requires a separately authorized adapter.

## Validation, compilation and execution

Validate the envelope, registered URI, argument types and constraints, allowed effects and exact contract digest. Canonical JSON uses UTF-8, sorted object keys, compact separators and rejects non-finite numbers. The contract digest excludes the derived `contract_digest` field itself and includes the public semantic and schema fields.

Compile validated calls deterministically to API records (`uri`, `args`, `expected_contract_digest`) or URI DSL (`uri: <stable-uri> <JSON arguments>`). Literal strings are data, including shell metacharacters. Never evaluate templates, generated Python or shell source to interpret a plan. Compilation reports `executed: false`.

Execution is a separate host boundary. Immediately before each call, revalidate the current contract, source snapshot, policy and arguments. A durable workflow adapter preserves order, dependency state and digest bindings; a compiler artifact alone is not proof of completed execution. A sequence requires an adapter with durable step validation. Read-only questions select inspection operations and never silently delegate development or mutation.

## Evaluation and deployment

Measure operation accuracy, argument accuracy, constraints, clarification, abstention, sequence order and p50/p95 latency separately. Report each required language independently. Gold cases include SVG→PNG and PNG→SVG, negation, typos, missing inputs, unsupported operations and multistep requests. Fixed-model tests must bind model revisions, catalog digests and runtime configuration. Synthetic conformance tests do not establish model accuracy.

Granite Embedding 97M Multilingual R2 and Qwen3.5-2B are candidates for measurement, not mandatory dependencies or proven production defaults. Other embedding/extraction/generation backends may implement the same contract. Do not claim universal multilingual parity from a model card.

Reference runtime owner: `paxlet-com/dockuri`, module `dockuri_nl`; consumers can inject its planner or use the bounded `dockuri nl` JSON stdin/stdout interface. The standard does not own a daemon. Runtime adoption and measured model quality remain distinct from publication of this specification.

Primary references: [Granite](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2), [Qwen](https://huggingface.co/Qwen/Qwen3.5-2B), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

Run reference schema conformance with `python -m pip install pytest jsonschema` and `python -B -m pytest standard/test_semantic_profile.py`. An explicit dependency skip is not conformance evidence.
