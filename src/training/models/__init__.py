"""Per-architecture workarounds, kept out of the backend code.

A module in here exists because one model does something the stack's generic
path gets wrong -- a renamed architecture string, an attention shape megatron's
kernels will not take, a field megatron-bridge spells two ways. The backends
(megatron_worker, fft_trainer_worker, ...) stay architecture-neutral and call
into here; nothing in here knows about open-rl's training loop.

Each is a monkeypatch against a pinned dependency, so each says which versions
it was verified against and what it costs if the seam moves.
"""
