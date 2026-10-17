"""Build docs/index.html (GitHub Pages) from the result files, so that every number on
the site comes from a logged run.   python docs/build_site.py"""
import json, html
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]

hae = json.load(open(R / "HAE/results/metro/results.json"))
fz = np.load(R / "VHAE/results/metro/forecasts_fig.npz", allow_pickle=True)
import glob
vm = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "VHAE/results/metro/results*.csv"))) if "_fig" not in f and "beta" not in f]).drop_duplicates(["method", "seed"])
vf = pd.read_csv(R / "VHAE/results/fertility/results.csv")

def path(y, x0, w, ymax, h):          # polyline points for a 168-hour strip
    n = len(y); return " ".join(f"{x0 + w * i / (n - 1):.1f},{h - h * v / ymax:.1f}" for i, v in enumerate(y))

truth, haef, lo, hi = (fz[k].reshape(-1) for k in ("truth", "HAE", "VHAE_lo", "VHAE_hi"))
W, H = 1000, 260; ymax = 1.08 * max(truth.max(), hi.max())
band = path(hi, 0, W, ymax, H) + " " + " ".join(reversed(path(lo, 0, W, ymax, H).split()))
days = [str(d) for d in fz["dates"]]
ticks = "".join(f'<line x1="{W*i/7:.1f}" x2="{W*i/7:.1f}" y1="0" y2="{H}" class="day"/>'
                f'<text x="{W*i/7+6:.1f}" y="{H-8}" class="lab">{pd.Timestamp(str(d)).strftime("%a %b %-d")}</text>'
                for i, d in enumerate(days))
hero_svg = f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="Seven held-out days of Metro traffic: observed curve, HAE point forecast and VHAE 90 percent band">
<polygon points="{band}" class="band"/>{ticks}
<polyline points="{path(truth,0,W,ymax,H)}" class="obs"/>
<polyline points="{path(haef,0,W,ymax,H)}" class="hae"/></svg>'''

def mstat(m, col):
    s = vm[vm.method == m][col]; return s.mean(), (s.std() if len(s) > 1 else float("nan"))
def fmt(m, col, d=3):
    a, b = mstat(m, col); return f"{a:.{d}f}" + ("" if np.isnan(b) else f" <span class=sd>({b:.{d}f})</span>")
rows = ""
for m, lab in [("HPCA", "HPCA + VAR"), ("HAE", "HAE + VAR"), ("VHAE", "VHAE + VAR")]:
    rows += (f"<tr><th scope=row>{lab}</th><td>{fmt(m,'fore_func')}</td><td>{fmt(m,'fore_nonfunc')}</td>"
             f"<td>{fmt(m,'func_coverage90',2)}</td><td>{fmt(m,'func_interval_score90',2)}</td><td>{fmt(m,'func_crps',3)}</td></tr>")
nseed = int(vm[vm.method == "VHAE"].seed.nunique())
red = lambda b, h: 100 * (b - h) / b
hae_f = red(hae["HPCA"]["fore_func"], hae["HAE"]["fore_func"]); hae_n = red(hae["HPCA"]["fore_nonfunc"], hae["HAE"]["fore_nonfunc"])
fv = vf[vf.method.isin(["FAE", "VFAE"])].groupby(["method", "beta"])[["fore", "coverage90", "crps"]].mean().reset_index()
frows = "".join(f"<tr><th scope=row>{r.method}{'' if r.method=='FAE' else f' (β = {r.beta:g})'}</th><td>{r.fore:.4f}</td><td>{r.coverage90:.2f}</td><td>{r.crps:.3f}</td></tr>" for r in fv.itertuples())
for k in ("FPCA(K=3)",):
    r = vf[vf.method == k].iloc[0]; frows = f"<tr><th scope=row>FPCA (K = 3)</th><td>{r.fore:.4f}</td><td>{r.coverage90:.2f}</td><td>{r.crps:.3f}</td></tr>" + frows

page = (R / "docs/template.html").read_text()
for k, v in dict(HERO=hero_svg, HAE_F=f"{hae_f:.0f}", HAE_N=f"{hae_n:.0f}", VROWS=rows, NSEED=str(nseed), FROWS=frows,
                 D0=pd.Timestamp(str(days[0])).strftime("%B %-d"), D1=pd.Timestamp(str(days[-1])).strftime("%B %-d, %Y")).items():
    page = page.replace("{{" + k + "}}", v)
(R / "docs/index.html").write_text(page); print("wrote docs/index.html")
