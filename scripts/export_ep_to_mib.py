import argparse
import json
from pathlib import Path

import numpy as np
from eap.graph import Graph  # comes with MIB repo
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--ep-run", required=True)  # e.g. runs/gpt2-ioi-ep
args = parser.parse_args()
run_dir = Path(args.ep_run)

# 1. load the pruned model through TransformerLens
model = HookedTransformer.from_pretrained(run_dir)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

# 2. blank Graph with canonical edge list
g = Graph.from_model(model)

# 3. read the log-α mask (either from .npz or the model itself)
mask = np.load(run_dir / "mask.npz")
for edge_name, loga in mask.items():  # keys already match "a0.h1->a3.h4<q>" etc.
    p = float(1 / (1 + np.exp(-loga)))  # σ(log α)
    g.edges[edge_name]["score"] = p
    g.edges[edge_name]["in_graph"] = False  # evaluator will decide k-sized circuits

# 4. dump to JSON
out_file = run_dir / "importances.json"
with open(out_file, "w") as f:
    json.dump(g.to_dict(), f, indent=2)
print("Wrote", out_file)
