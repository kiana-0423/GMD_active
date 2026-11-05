"""DFT labeling.

- :class:`~gmd_active_learning.labeling.service.LabelingService`: the API the workflow calls
  (``prepare(region) -> LabelTask``, ``submit``, ``poll``, ``reconcile``, ``cancel``,
  ``collect -> LabelResult``), persistent and idempotent.
- :class:`~gmd_active_learning.labeling.retry_policy.RetryPolicy`: bounded per-failure-class retries.
- Legacy batch writers: ``CP2KLabeler`` (real CP2K inputs) and ``VASPLabeler`` (UNVALIDATED).

Submodules are imported on demand to keep this package light.
"""
