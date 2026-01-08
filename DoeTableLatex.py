from pprint import pprint
import math

def _combine_stats(items):
    """
    items: lista di tuple (n, it30, mean, std)
    ritorna: (N, it30, mean_all, std_all) con std campionaria combinata
    """
    N = sum(n for n, _, _, _ in items)
    it30 = sum(it for _, it, _, _ in items)
    if N == 0:
        return 0, 0, 0.0, 0.0

    mean_all = sum(n * m for n, _, m, _ in items) / N

    if N == 1:
        return N, it30, mean_all, 0.0

    # varianza campionaria combinata
    ss = 0.0
    for n, _, m, s in items:
        if n <= 1:
            ss += n * (m - mean_all) ** 2
        else:
            ss += (n - 1) * (s ** 2) + n * (m - mean_all) ** 2

    var_all = ss / (N - 1)
    std_all = math.sqrt(max(var_all, 0.0))
    return N, it30, mean_all, std_all


def stats_to_latex_table(stats, versions=("v1", "v2", "v3"), decimals=2, caption=None, label=None):
    """
    stats: dict come il tuo
    genera una tabella LaTeX con colonne raggruppate per app e subcolonne v1 v2 v3 All.
    Righe: N tests, iteration=30, %failed, media iter, std iter
    """
    apps = list(stats.keys())

    # precompute "All" per app
    per_app = {}
    for app in apps:
        rows = []
        for v in versions:
            n, it30, mean, std = stats[app][v]
            rows.append((n, it30, mean, std))
        per_app[app] = {v: stats[app][v] for v in versions}
        per_app[app]["All"] = list(_combine_stats(rows))

    def fmt_num(x):
        return f"{x:.{decimals}f}"

    def fmt_pct(num, den):
        if den == 0:
            return "0.00\\%"
        return f"{(100.0 * num / den):.{decimals}f}\\%"

    # header
    col_spec = "l|" + "|".join(["cccc"] * len(apps))
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append(
        " & " + " & ".join([f"\\multicolumn{{4}}{{c}}{{{app}}}" for app in apps]) + " \\\\"
    )
    lines.append("\\hline")
    lines.append(
        "Index & " + " & ".join(["v1 & v2 & v3 & All"] * len(apps)) + " \\\\"
    )
    lines.append("\\hline")

    # rows
    row_names = [
        ("N tests", lambda n, it30, mean, std: str(int(n))),
        ("iteration=30", lambda n, it30, mean, std: str(int(it30))),
        ("%failed", lambda n, it30, mean, std: fmt_pct(it30, n)),
        ("media iter", lambda n, it30, mean, std: fmt_num(mean)),
        ("std iter", lambda n, it30, mean, std: fmt_num(std)),
    ]

    for title, fn in row_names:
        row = [title]
        for app in apps:
            for v in (*versions, "All"):
                n, it30, mean, std = per_app[app][v]
                row.append(fn(n, it30, mean, std))
        lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)

    
