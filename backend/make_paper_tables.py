"""Generate LaTeX tables for the paper directly from results/results.json and results/extra.json (no hand-typed numbers)."""
import json
d, x = json.load(open("results/results.json")), json.load(open("results/extra.json"))
S = d["summary"]
cols = [("RMSE", "RMSE", False), ("MAE", "MAE", False), ("Precision@10", "P@10", True),
        ("Recall@10", "R@10", True), ("NDCG@10", "NDCG@10", True), ("HitRate@10", "HR@10", True)]
best = {}
for k, _, hi in cols:
    vals = {n: v[k]["mean"] for n, v in S.items() if k in v}
    best[k] = (max if hi else min)(vals, key=vals.get)
rows = []
for n, v in S.items():
    cells = []
    for k, _, _ in cols:
        if k not in v: cells.append("--"); continue
        c = f"{v[k]['mean']:.3f}\\,$\\pm$\\,{v[k]['std']:.3f}"
        cells.append(f"\\textbf{{{c}}}" if best[k] == n else c)
    rows.append(f"{n} & " + " & ".join(cells) + " \\\\")
t1 = ("\\begin{table*}[t]\n\\centering\n\\caption{Test-set results (mean\\,$\\pm$\\,std over three random 80/20 per-user splits). "
      "Rating metrics: lower is better. Ranking metrics at $K{=}10$ (relevant = held-out rating $\\geq 4$): higher is better. Best per column in bold.}\n"
      "\\label{tab:main}\n\\small\n\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\nModel & " + " & ".join(h for _, h, _ in cols)
      + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
ab = ["Item-KNN, similarity-sum score", "Item-KNN, predicted-rating score", "SVD, predicted-rating score"]
arows = []
for n in ab:
    for suf in ("", " + min 20 ratings"):
        v = x[n + suf]
        arows.append(f"{n.replace(', ', ' / ')}{'' if not suf else ' + support filter'} & {v['Precision@10'][0]:.3f} & {v['NDCG@10'][0]:.3f} & {v['HitRate@10'][0]:.3f} \\\\")
t2 = ("\\begin{table}[t]\n\\centering\n\\caption{Effect of the ranking score and of a minimum-support filter (candidates with $\\geq 20$ training ratings). Mean over three splits.}\n"
      "\\label{tab:ablation}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccc}\n\\toprule\nVariant & P@10 & NDCG@10 & HR@10 \\\\\n\\midrule\n"
      + "\n".join(arows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open("paper/tables.tex", "w").write(t1 + "\n" + t2)
print(t1); print(t2)
