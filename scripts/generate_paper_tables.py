import pandas as pd
import glob
from pathlib import Path

def load_summaries():
    files = glob.glob("data/eval_results/summary_tier1_*.csv")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def main():
    df = load_summaries()
    
    print("\n" + "="*80)
    print("TABLE 1: MAIN PERFORMANCE COMPARISON (Tier 1)")
    print("="*80)
    
    # Get the full_system profile rows
    main_df = df[df["profile_name"] == "full_system"].copy()
    
    # We might have duplicates if run multiple times, take the latest
    main_df = main_df.drop_duplicates(subset=["mode"], keep="last")
    
    cols = ["mode", "rouge_l_mean", "bertscore_f1_mean", "nar_mean", "hr_mean", "acr_mean", "swos_mean", "ndi_mean"]
    print(main_df[cols].to_markdown(index=False))
    
    print("\n" + "="*80)
    print("TABLE 2: ABLATION ANALYSIS (Finetuned Mode, Tier 1)")
    print("="*80)
    
    ft_df = df[df["mode"] == "finetuned"].copy()
    ft_df = ft_df.drop_duplicates(subset=["profile_name"], keep="last")
    
    abl_cols = ["profile_name", "fluency_score_mean", "safety_score_mean", "hr_mean", "acr_mean"]
    print(ft_df[abl_cols].to_markdown(index=False))

if __name__ == "__main__":
    main()