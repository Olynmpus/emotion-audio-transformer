import pandas as pd
from scipy.stats import mannwhitneyu

path = r"outputs_wdrc_eval\paired_acoustic_changes.csv"
df = pd.read_csv(path)

measures = [
    "rms_change_db",
    "envelope_SD_change",
    "crest_factor_change",
]

for condition in df["condition"].unique():
    sub = df[df["condition"] == condition]
    harmed = sub[sub["outcome"] == "CorrectToWrong"]
    stable = sub[sub["outcome"] == "BothCorrect"]

    print(f"\n{condition}: harmed n={len(harmed)}, both-correct n={len(stable)}")
    for col in measures:
        x = harmed[col].dropna()
        y = stable[col].dropna()
        if len(x) and len(y):
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            print(
                f"  {col}: harmed median={x.median():.3f}, "
                f"both-correct median={y.median():.3f}, p={p:.4f}"
            )
