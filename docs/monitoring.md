# Online monitoring, events and hotspots

This page covers `gmd_active_learning.monitors`, `gmd_active_learning.events` and
`gmd_active_learning.hotspots`. Together they turn a stream of MD frames into per-atom anomaly
scores, frame actions, events with pre- and post-trigger context, and spatial hotspot clusters.
The design follows PHAL's atom-wise staged monitor (`hotspot_al.monitor`, commit 96263b6). The
deliberate differences are listed at the end.

Units: positions in Å, forces in eV/Å, time in fs. Atoms are identified by stable atom IDs
(`Frame.atom_ids`), never by their position in an array.

## Quick start

```python
from gmd_active_learning.monitors import OnlineMonitorDriver
from gmd_active_learning.events import EventStore

driver = OnlineMonitorDriver(
    cfg["monitor"], cfg["events"], cfg["hotspots"],   # sections from configs/sections/*.yaml
    reference_forces=reference_model,                  # optional: callable(Frame) -> (N, 3) eV/A
    on_event=EventStore("run/events"),                 # or any callable(EventRecord)
)

# Callback mode (the GMD adapter drives MD):
result, finished_events = driver.process(frame)        # result.action: continue/save/checkpoint/stop
...
driver.save_state("run/monitor_state.json")            # end of a segment: the open event survives
driver.load_state(json.load(open("run/monitor_state.json")))  # next segment
driver.finish("end_of_stream")                         # end of run: flush the open event

# Loop mode:
summary = driver.run(frames)                           # an iterable of Frame or a FrameSource
```

The monitor never starts an inference backend. Re-evaluated reference forces come from the
caller's `ReferenceForceProvider`, a callable that takes a `Frame` and returns forces of shape
(N, 3). Committee forces come from `Frame.committee_forces`, or from an optional
`CommitteeForceProvider` when the frame has none.

## Pipeline per frame

1. **Light metrics** are computed on every frame, so that the temporal references and the hard
   STOP rules always stay current.
2. If `stages.light_interval` is not due, the frame's stage is `SKIPPED`. Nothing is scored and
   the running statistics are not updated.
3. If `stages.scheduled_full_interval` is k > 1 and `frame_index % k == 0`, every metric is
   evaluated and the frame is scored at the `FULL` stage.
4. Otherwise the frame is scored at the `LIGHT` stage.
   - It escalates to `PHYSICS` when `stages.physics_interval` is due. The default, `triggered`,
     means the light stage triggered. The physics stage adds the local LJ residual and the MLIP
     force deviation.
   - It escalates further to `FULL` when `stages.full_interval` is due (by default, the physics
     stage triggered). The full stage adds the committee deviation.
5. The final stage's scores give the result.
   - `triggered`: at least `min_trigger_atoms` atoms score at or above that stage's threshold
     (screen, physics or label).
   - `hotspot_atom_ids`: atoms with a score at or above `label_threshold`. If the frame
     triggered but no atom reaches `label_threshold`, the atoms with the maximum score are used.
   - `trigger_reasons`: metrics ranked by their mean unweighted score over the hotspot atoms.
6. The running statistics are updated once, using only the atoms that were evaluated.
7. The `FrameAction` is chosen by precedence, `stop` > `checkpoint` > `save` > `continue`:
   - `save` if the frame triggered and `save_on_trigger` is set
   - `checkpoint` if `max_score` ≥ `checkpoint_score`
   - `stop` if `max_score` ≥ `stop_score`, any `rmin` < `stop_min_distance`, or any |F| >
     `stop_max_force`

   The rules that fired are listed in `metadata.action_reasons`.

`frame_index` counts the frames processed, not MD steps. It is persisted in the state.

## Metrics

| Metric | Value per atom | Stage | Score |
|---|---|---|---|
| `force` | \|F_i\| (eV/Å) | light | gated z: z if z ≥ `z_threshold`, else 0 |
| `delta_force` | \|F_i(t) − F_i(t_prev)\| | light | gated z |
| `displacement` | \|MIC(r_i(t) − r_i(t_prev))\| (Å), exact minimum image, including skewed and partially periodic cells | light | gated z |
| `rmin` | nearest-neighbour distance (Å). Atoms with no neighbour report the neighbour cutoff. | light | max(0, (thr − rmin)/thr). `thr: null` means the value is measured but not scored. |
| `delta_q` | \|q_i(t) − q_i(t_prev)\|, with q_i = Σ_j 1/(1 + (r_ij / (scale·(R_i+R_j)))^power) for r_ij ≤ the coordination cutoff | light | max(z, max(0, (\|dq\| − thr)/thr)) |
| `lj_residual` | local LJ projection residual (dimensionless; see below) | physics | max(residual, z) on fitted atoms, 0 on atoms skipped by lazy evaluation |
| `mlip_force_deviation` | \|F_MD − F_reference\| (eV/Å) | physics | z |
| `committee` | sqrt(mean_m \|F^m − mean F\|²) (eV/Å) | full | max(raw, z) |

The z-score is z = \|x − mean\| / max(std, `min_std`), computed from running statistics pooled
over all atoms of every frame that updated them (Chan's parallel update, no decay).

- Warmup: a metric's z-score is used only after `warmup_frames` stats-updating frames. Before
  that, `warmup_policy: suppress` (the default) drops the z-based parts of the score.
  `warmup_policy: raw` reproduces PHAL.
- `running_stats.enabled: false` replaces z with the raw \|x\|.

Every metric reports a `MetricStatus`:

- `MEASURED`, with an `evaluated` mask per atom
- `NOT_EVALUATED`, with a reason: `disabled`, `not escalated beyond light`, `light_interval`,
  `lazy: no suspicious atom` or `no previous frame`
- `UNAVAILABLE`: no committee, a committee with fewer than `min_members` members, or no reference
  force provider

Unevaluated and unavailable metrics contribute exactly 0 and never enter the statistics.

### Local LJ residual (sign-corrected)

For atom i with neighbours j within `lj_cutoff`, let d_ij = r_j − r_i. The fitted force is
F_i ≈ α a_i + β b_i, where a_i = Σ −2 d/r¹⁴, b_i = Σ d/r⁸, α = 24εσ¹² and β = 24εσ⁶. This is
the physical force, F = −∇E.

- The fit is a 3×2 least-squares solve on column-normalised design.
- A fit is valid only if all of these hold: rank 2, condition number ≤ `condition_limit`, α > 0,
  β > 0, and σ in [`sigma_min`, `sigma_max`].
- The residual is \|F_fit − F_i\| / (\|F_i\| + δ). Invalid fits report a residual of 1.0.
- Lazy evaluation fits only the atoms whose weighted light score is ≥ `lazy_threshold`. If no
  atom qualifies, it fits the single argmax atom, provided its score is > 0.
- `estimator: nonlinear` selects a bounded scipy reference fit, for diagnostics only. Its residual
  scale differs from the linear fit's.
- `monitors.lj_projection.fit_lj_projection(mode="global_pair" | "local_atom")` exposes both LJ
  modes behind one sign convention.

### Engines

| `monitor.engine` | What it does |
|---|---|
| `staged` (default) | The atom-wise monitor described above (`StagedOODMonitor`). |
| `global_lj` | GMD's original frame-level `ReliabilityMonitor`: a windowed per-species-pair LJ fit, frame-global physical checks and committee deviation, mapped onto `MonitorResult`. |

With `global_lj`:

- SAFE, CANDIDATE and STOP map to `triggered` and to the SAVE and STOP actions.
- `max_score` is the frame risk score.
- `atom_scores` are the per-atom LJ fit residuals of the current frame, in eV/Å.
- The hotspot atoms are the atoms with the largest residual. This engine cannot localise anomalies.

## Events

`EventBuffer` keeps the last `pre_trigger_frames + 1` frames. This depends on PHAL's ordering:
`capture` runs before `push`, and a trigger frame that was already pushed is excluded from its own
pre-context.

- A trigger opens an event. The next `post_trigger_frames` frames become its post-context.
- A trigger while an event is open is merged into it without extending the window, as in PHAL:
  - `trigger_steps` grows
  - hotspot atoms are unioned
  - per-atom scores take the maximum
  - reasons are de-duplicated
- `event_id` is `stable_id("evt", run_id, first_trigger_step)`, so it is reproducible.
- An event finalised early has `complete=False` and a `truncation_reason`. The reason is one of
  `end_of_stream`, `stopped`, `error`, `timeout`, `max_frames` or `interrupted`.
- `cooldown_steps`: after an event's last frame, triggers at steps before
  `last + cooldown_steps` open no event. They are counted in `counters["suppressed_by_cooldown"]`.
- Finished events are not retained. Only `retain_recent` summaries are kept in memory.
- `EventStore` / `write_event` write `<root>/<event_id>/`:
  - `frames.npz`: bit-exact arrays
  - `frames.extxyz`: a human-readable copy
  - `event.json`: written last and atomically

  `read_event` rebuilds the `EventRecord`.

## Hotspots

`hotspots_from_result(frame, result, cfg)` clusters `result.hotspot_atom_ids` by single linkage.
Atoms whose minimum-image distance is ≤ `merge_radius` are merged; a distance exactly equal to
`merge_radius` merges.

- Each hotspot carries its stable atom IDs, a minimum-image centroid anchored at its smallest atom
  ID, `max_score`, the trigger reasons and a deterministic `hotspot_id`.
- Members and clusters are sorted by atom ID, and hotspots by (−max_score, smallest atom ID), so
  the result does not depend on the engine's atom order.
- The driver stores the clusters of each triggered frame in `result.metadata["hotspots"]`.
- `detect_hotspots(frame, scores, threshold=...)` is the PHAL-style entry point that works from
  scores.

## Driver, sources and state

`OnlineMonitorDriver.run(frames)` accepts any iterable of `Frame` or a `FrameSource`.

- `FrameSource.next_frame(timeout)` returns a `Frame`, `None` (end of stream) or `POLL_TIMEOUT`.
  After `driver.max_consecutive_timeouts` timeouts in a row, the run ends with reason `timeout`.
- Per-frame exceptions follow `driver.continue_on_error` and `driver.max_errors`.
- A STOP action ends the run when `driver.stop_on_stop_action` is set.
- Progress goes to the `on_progress` callback, and to `driver.progress_file` (written atomically).

`to_state()` and `load_state()` round-trip everything needed to resume a segmented run, as
JSON-serialisable data:

- frame index
- running statistics
- previous positions, forces and coordination numbers, keyed by atom ID
- event history
- the open event
- cooldown

A segmented run with a state round trip reproduces the continuous run exactly; this is tested.

## Configuration

The defaults are in `configs/sections/monitor.yaml`, `events.yaml` and `hotspots.yaml`. The models
are `MonitorConfig`, `EventsConfig` and `HotspotsConfig`. Unknown keys are rejected.

Every metric can be switched off with `metrics.<name>.enabled: false`. Other switches:

| Setting | Effect |
|---|---|
| `stages.physics_interval: never` | Turns off the LJ and MLIP stage. |
| `stages.full_interval: never` | Turns off the committee stage. |
| `monitor.enabled: false` | Every frame is `SKIPPED` and nothing is computed. |
| `events.enabled: false` | No events are captured. |
| `hotspots.enabled: false` | No hotspot clusters are computed. |

`monitor_config_from_phal(phal_cfg)` and `events_config_from_phal(phal_cfg)` translate a PHAL
configuration:

| PHAL key | GMD_active key |
|---|---|
| `monitor.{force,delta_force,displacement}_z_threshold` | `metrics.<m>.z_threshold` |
| `monitor.rmin_threshold` | `metrics.rmin.threshold` |
| `monitor.delta_q_threshold` / `coordination_scale` / `coordination_smoothing_power` | `metrics.delta_q.threshold` / `scale` / `power` |
| `monitor.lj_cutoff` / `coordination_cutoff` | `neighbors.*` |
| `monitor.light_interval` / `physics_interval` / `committee_interval` | `stages.light_interval` / `physics_interval` / `full_interval` |
| `online.monitor_freq` | `stages.scheduled_full_interval` |
| `ood_score.weights.<m>` | `metrics.<m>.weight` |
| `ood_score.{screen,physics,label}_threshold`, `min_trigger_atoms` | `stages.*` |
| `ood_score.lj_lazy_threshold` | `metrics.lj_residual.lazy_threshold` |
| `ood_score.running_stats.*` | `running_stats.*` |
| `buffer.pre_trigger_frames` / `post_trigger_frames` / `maxlen` | `events.*` |
| `hotspot.merge_radius` | `hotspots.merge_radius` |
| `hotspot.threshold` | removed (see F-16 below) |
| `online.continue_on_error` / `max_errors` / `summary_interval` | `driver.*` |

## Deviations from PHAL

| Item | PHAL | GMD_active |
|---|---|---|
| F-01: LJ sign | Fitted the negated force, so physical LJ forces scored residual 1.0. | Fits F = −∇E. Exact ASE LJ forces give a residual below 1e-8 and recover ε and σ. |
| F-06: missing committee | Zero-filled, which reads as perfect agreement and can produce a spurious z(0). | `UNAVAILABLE`. Scores 0 and stays out of the statistics. A 1-member committee is also unavailable. |
| F-07: truncated events | No marker. `RollingBuffer.events` grew without bound. | `complete=False` with a `truncation_reason`. Only bounded summaries are kept. |
| F-15: event IDs | `evt-<step>-<uuid4>`. | `stable_id("evt", run_id, first_trigger_step)`. |
| F-16: LJ cost | Fitted twice per escalated frame (21 fits for ethane_box's 13 escalated frames; reproduced against the reference tree). | Fitted once per escalated frame that has a suspicious atom (12 fits). |
| F-16: warmup | Raw eV/Å values compared against z thresholds. | `warmup_policy: suppress` (default). `raw` is available for replays. |
| F-16: `rmin_threshold` | Default `null`, so close contacts were never scored. | Default 0.8 Å. A contact contributes at most `weight × 1`, so the hard rule `actions.stop_min_distance` (0.6 Å) is the safety net. |
| F-16: coordination keys | `coordination_scale` and `coordination_smoothing_power` were validated but never read. | Honoured as `metrics.delta_q.scale` and `power`. |
| F-16: hotspot thresholds | Two: `ood_score.label_threshold` and `hotspot.threshold`. A light trigger could leave zero clusters. | One: `stages.label_threshold`. Clustering uses the monitor's hotspot atoms. |
| First-frame deltas | Measured zeros, which biased the baseline. | `NOT_EVALUATED` (default). `first_frame_deltas: zero` is available for replays. |
| Invalid interval strings | Silently treated as `triggered`. | Configuration error. |
| `rmin` with no neighbour | Exact distance below 512 atoms, the cutoff at or above 512. | Always the neighbour cutoff. Thresholds must be below the cutoff. |
| Temporal state | Indexed by array position. | Aligned by atom ID. A changed atom set restarts the deltas. |
| Frame actions, cooldown, state | None. | `FrameAction` policy, `cooldown_steps`, and `to_state`/`load_state`. |

### Parity with the golden fixtures

The parity tests are in `tests/monitors/test_phal_parity.py`. They replay
`tests/fixtures/phal_reference/<scenario>/input_frames.extxyz` with `config.json` translated by
`monitor_config_from_phal` (legacy scoring conventions). The MLIP reference is committee member 0.

With the corrected LJ estimator fed the **negated** forces (corrected(−F) = PHAL(F)), every frame
of all three scenarios matches PHAL. This covers stage, triggered, hotspot atoms, ordered reasons,
`max_score`, `lj_fit_count`, every per-atom metric score and the atom scores, to within 1e-4
relative. The tolerance reflects the 8-decimal rounding of the stored inputs. All events match:
steps, pre/post steps, `trigger_steps`, hotspot atoms, reasons and scores.

With the **physically correct** sign and otherwise PHAL conventions, only the LJ residual changes.
All other metric scores still match. The resulting differences:

| Scenario | Frames whose hotspot atoms or reasons change | Trigger changes | Events |
|---|---|---|---|
| ethane_box | 5, 6, 7, 8, 13 | none | identical |
| polyethylene_chains | 5, 7, 8, 9, 11 | frame 7 (step 70) now triggers; frame 8 (step 80) no longer does | first event's `trigger_steps` [50, 60, 80] → [50, 60, 70] |
| pt111_co_slab | 6, 7, 11, 12 | none | identical |

These are expected. The synthetic MD forces are harmonic plus noise, not LJ, so roughly half of the
fitted atoms are "valid" under either sign convention. Correcting the sign changes which atoms get
residual 1.0.

With the full GMD_active defaults (`warmup_policy: suppress`, `first_frame_deltas: not_evaluated`),
three more things change. The warmup frames lose their raw-value contributions. Some hotspot sets
on triggered frames shrink or grow. The z-baselines shift slightly, because the first frame no
longer adds zero deltas. The stage of every frame is still unchanged, the trigger changes are the
same two polyethylene frames, and the events are those of the table above. This is pinned by
`test_full_defaults_change_no_stage_and_no_event`.

## Limitations

- The local LJ residual has one residual degree of freedom per atom (3 force components, 2
  parameters). It flags forces that are not radial-LJ-like but cannot detect wrong pair strengths.
- Running statistics are pooled over all species and never decay. Physics- and full-stage metrics
  accumulate only on escalated frames, so their baseline is biased toward suspicious frames, as in
  PHAL.
- `committee` and `lj_residual` scores mix raw values with z-scores (`max(raw, z)`), as in PHAL.
- `mlip_force_deviation` is a pure z-score. If the engine runs the same model, it measures
  precision drift; tune `running_stats.min_std` accordingly.
- `displacement` folds jumps longer than half a cell period back to the minimum image.
- Hotspot centroids of clusters wider than half a cell period are folded.
