# Day 5 execution checkpoint

The source and configuration were committed at `de419ad916e57e04271c14fd830a8a52022b764e` before any Day 5 GPU predictions. Source digest: `fea39b4e91e3e02177503b1f3c0444c54f62966e8cf4a9844ae76833cbfc6314`.

The research Colab has a fresh Tesla T4 runtime. Cell 18 runs the three smoke clips. Cell 19 queues the unchanged twelve-clip test after validating successful smoke completion. Cell 20 checks completeness and creates a checksum-inventoried result archive and a visual comparison. No model outcome is claimed at this checkpoint.

Notebook: https://colab.research.google.com/drive/1wipvFdtlDOuRwp6Cd9sQpQUCZQNxo2HP

Runtime output: `/content/day5_parent_run`. The runner checks source, input, prompt, checkpoint and cache hashes before reusing completed clips. The new notebook `Parent_Constraint_Colab.ipynb` gives clean reproduction instructions.

Remaining work: finish all fixed clips, download and independently audit the cached masks and metrics, commit results and interpretation, then preserve the executed notebook. Do not modify the policy in response to test outcomes.
