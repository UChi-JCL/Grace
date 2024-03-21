import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

def read_df(path):
    if os.path.exists(path):
        return pd.read_csv(path).query("frame_id < 16")
    else:
        print("Skip the file because", path, "does not exist!")
        return None

def quick_plot_size(df, label):
    if df is None:
        print("Skip", label, "during size plot because dataframe is None")
        return

    df = df.sort_values("size").reset_index(drop=True)
    if "ssim_db" not in df.columns:
        df["ssim_db"] = -10 * np.log10(1 - df["ssim"])
    plt.plot(df["size"], df["ssim_db"], label=label)

def quick_plot_loss(df, label):
    if df is None:
        print("Skip", label, "during loss plot because dataframe is None")
        return
    df = df.sort_values("loss").reset_index(drop=True)
    if "ssim_db" not in df.columns:
        df["ssim_db"] = -10 * np.log10(1 - df["ssim"])
    plt.plot(df["loss"], df["ssim_db"], label=label)

def interpolate_quality(df, target_size):
    """
    assume input df has loss, size, ssim
    """
    if df is None:
        print("Skip", label, "during quality interpolation because dataframe is None")
        return
    df = df.sort_values(['loss', 'size'])
    def group_interpolate(group):
        return pd.Series({'ssim': np.interp(target_size, group['size'], group['ssim'])})

    result = df.groupby(["loss"]).apply(group_interpolate).reset_index()
    result["ssim_db"] = -10 * np.log10(1 - result["ssim"])
    return result

def quick_plot_fec(df, target_size, fec_ratio):
    if df is None:
        print("Skip", label, "during fec plot because dataframe is None")
        return
    idf = interpolate_quality(df.groupby(["loss", "model"]).mean().reset_index(), size * (1-fec_ratio))
    quality = float(idf["ssim_db"])
    x = [0, fec_ratio - 0.01, fec_ratio + 0.01]
    y = [quality, quality, 8]
    plt.plot(x, y, label=f"{fec_ratio*100:.1f}% FEC")

video_filter = "video.str.contains('game')"
df_grace = read_df("grace/all.csv").query(video_filter)
print(df_grace["video"].unique())
df_265 = read_df("h265/all.csv").query(video_filter)
df_264 = read_df("h264/all.csv").query(video_filter)
df_pretrain = read_df("pretrained/all.csv").query(video_filter)
df_error = None #read_df("error_concealment/all.csv").query(video_filter)

''' QUALITY VS SIZE CURVE '''
fig = plt.figure()
if df_grace is not None:
    quick_plot_size(df_grace.query("nframes == 0").groupby("model_id").mean().reset_index(), "grace")
if df_pretrain is not None:
    quick_plot_size(df_pretrain.query("nframes == 0").groupby("model_id").mean().reset_index(), "pretrained")
if df_265 is not None:
    quick_plot_size(df_265.groupby("model").mean().reset_index(), "265")
if df_264 is not None:
    quick_plot_size(df_264.groupby("model").mean().reset_index(), "264")
plt.xlim(0, 30000)
plt.grid()
plt.legend()
fig.savefig("ssim_size.png")


''' QUALITY VS LOSS CURVE '''
size = float(df_grace.query("nframes == 1 and model_id == 4096").mean()["size"])
for nframes in [1,3,5]:
    fig = plt.figure()
    if df_grace is not None:
        quick_plot_loss(interpolate_quality(df_grace.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size), "grace")
    if df_error is not None:
        quick_plot_loss(interpolate_quality(df_error.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size), "error concealment")
    if df_pretrain is not None:
        quick_plot_loss(interpolate_quality(df_pretrain.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size), "pretrained")
    if df_265 is not None:
        quick_plot_fec(df_265, size, 0.2)
        quick_plot_fec(df_265, size, 0.5)
    if df_264 is not None:
        quick_plot_fec(df_264, size, 0.2)
        quick_plot_fec(df_264, size, 0.5)
    plt.grid()
    plt.legend()
    fig.savefig(f"ssim_loss-{nframes}.png")
