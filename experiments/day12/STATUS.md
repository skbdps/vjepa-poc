# Day12 status: actual video comparison completed

All 20 Wan/VACE video renders completed on Kaggle at **2026-10-10 23:15:04 UTC**.
Assessment scored all 20 clips with no problems. All 60 RGB, latent and GIF
payloads were downloaded and hash-verified. Exact guide, source and prompt
bindings also passed. No render error files were present.

**Conclusion:** the pipeline controls the commanded motion on these two
procedural development scenes. The learned CNN translator adds no demonstrated
benefit over the matched `no_f` appearance-transport baseline: the baseline has
lower position, moved-object texture and source-hole error in all four matched
cases. See [RESULTS.md](RESULTS.md) for comparisons and limitations.

One Tesla T4 was used from the T4 x2 allocation. Nine-frame 384×384 clips took
31.59–37.83 seconds each after setup, with at most 7.230 GiB recorded per-clip
PyTorch CUDA allocation. The render span was 12m12.8s; download and prompt
encoding are additional. No paid generation API was used. After the saved snapshot and downloaded results
were verified, the idle session was stopped to release GPU quota.

## Preserved evidence

- `run_kaggle/assessment.json` and `render_metrics.json`: full metrics and bindings.
- `run_kaggle/summary/`: all 20 rows, 16 matched comparisons and four labeled GIFs.
- `run_kaggle/evidence/results/`: lossless archive of all 71 exported files,
  including 20 RGB arrays, 20 generated latent arrays, 20 GIFs and prompt caches.
- `run_kaggle/notebook_completed.ipynb`: downloaded executed notebook.
- `run_kaggle/saved_outputs.jpg`: private Kaggle version with saved outputs.
- `run_recovered/evidence/`: original guides and earlier video-VAE checks.

Kaggle version 1 was saved as a **Quick Save with outputs**, not a rerun:
https://www.kaggle.com/code/skbdps/vjepa-day12-gpu-preflight/output?scriptVersionId=357185481

## Access and interruption history

Colab sign-in succeeded but GPU allocation was refused due to usage limits.
The user completed Kaggle phone verification, enabling GPU and Internet.
Actual Kaggle preflight passed at 22:55 UTC.

During inference an editor autosave conflict required notebook backups and a
refresh. The same kernel continued rendering during the long reconnection delay;
it was not restarted. The editor reconnected after completion. Some earlier cell
execution counters and outputs were lost from the draft. The final notebook's
completion, assessment and export cells plus the hashed raw results establish
completion; intermediate notebook snapshots preserve the earlier execution.

Earlier workspace maintenance removed uncommitted Day12 drivers. Original guide
and video-VAE output bytes survived and were independently verified; continuation
drivers are explicitly reconstructed versions. Generation used clean checkpoint
`eeaa0f946317f4313c76063dc17034fc0f1132f2` and those exact guides. This does not
establish completion of the older Day11 104-image batch, whose saved notebook
showed 43 outputs. Historical status is retained in
`run_kaggle/status_before_completion.md`; its pending-state statements are obsolete.

## Next scientific step

Separate the CNN's correction of moved tiles from its filling of newly exposed
background, scoring four combinations before another GPU batch. Test edited-state
training and region-balanced losses on fresh scenes. Temporal JEPA prediction,
face identity, stable object parts, physics and production quality remain untested.
