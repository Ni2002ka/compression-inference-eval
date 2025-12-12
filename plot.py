import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator
import numpy as np

def smooth_line(x, y, num=200):
    """Monotone smooth curve for RD points."""
    x_sorted = np.sort(x)
    y_sorted = np.array(y)[np.argsort(x)]
    f = PchipInterpolator(x_sorted, y_sorted)
    xs = np.linspace(min(x_sorted), max(x_sorted), num)
    ys = f(xs)
    return xs, ys




def plot_accuracy_by_dataset(df):
    plt.figure(figsize=(12,6))
    g = sns.catplot(
        data=df,
        x="compression",
        y="test_accuracy",
        hue="model",
        col="dataset",
        kind="bar",
        height=5,
        aspect=1
    )
    g.set_xticklabels(rotation=45)
    g.fig.suptitle("Accuracy vs Compression (per Dataset)", y=1.05)
    plt.show()


def plot_training_speed_by_dataset(df):
    plt.figure(figsize=(12,6))
    g = sns.catplot(
        data=df,
        x="compression",
        y="avg_epoch_time",
        hue="model",
        col="dataset",
        kind="bar",
        height=5,
        aspect=1
    )
    g.set_xticklabels(rotation=45)
    g.fig.suptitle("Avg training epoch time vs Compression (per Dataset)", y=1.05)
    plt.savefig("results/training_speed.png")


def plot_inference_speed_by_dataset(df):
    plt.figure(figsize=(12,6))
    g = sns.catplot(
        data=df,
        x="compression",
        y="test_time",
        hue="model",
        col="dataset",
        kind="bar",
        height=5,
        aspect=1
    )
    g.set_xticklabels(rotation=45)
    g.fig.suptitle("Inference Latency vs Compression (per Dataset)", y=1.05)
    plt.savefig("results/inference_speed.png")



def plot_heatmaps_by_dataset(df, metric="test_accuracy"):
    # x axis is compression, y axis is model
    # We generate one heatmap per dataset
    datasets = df["dataset"].unique()

    for ds in datasets:
        df_ds = df[df["dataset"] == ds]

        # Create a pivot table: rows=compression, columns=model, values=test_accuracy
        pivot = df_ds.pivot_table(
            index="compression",
            columns="model",
            values=metric
        )

        cmap = "crest" if metric != "test_accuracy" else "RdYlGn"
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            cmap=cmap,  
            fmt=".3f",
            linewidths=.5,
            cbar_kws={"label": metric}
        )

        plt.title(f"{metric} Heatmap — {ds}", fontsize=16)
        plt.xlabel("Model")
        plt.ylabel("Compression")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"results/{ds}_{metric}.png")



from matplotlib.lines import Line2D

def plot_rate_distortion(df, dataset_name, model_name):

    df_ds_model = df[(df["dataset"] == dataset_name) & (df["model"] == model_name)]
    sns.set_theme(style="whitegrid", context="talk")

    # Define groups
    group_webp = ["webp_1", "webp_5", "webp_20"]
    group_jpeg = ["jpeg_1", "jpeg_10", "jpeg_50"]
    group_neutral = ["png", "none"]

    # Trend line colors
    COLOR_WEBP = "#1f77b4"   # blue
    COLOR_JPEG = "#ff7f0e"   # orange

    # Scatter color palettes 
    webp_palette   = ["#6d86f7", "#2743c2", "#0a2391"]
    jpeg_palette   = ["#eda05c", "#cf6d17", "#7a3b02"]
    neutral_palette = ["#a8a8a8", "#000000"]

    # Build color map
    color_map = {}
    for c, col in zip(group_webp, webp_palette):
        color_map[c] = col
    for c, col in zip(group_jpeg, jpeg_palette):
        color_map[c] = col
    for c, col in zip(group_neutral, neutral_palette):
        color_map[c] = col

    df_ds_model["color"] = df_ds_model["compression"].map(color_map).fillna("#999999")

    df_webp = df_ds_model[df_ds_model["compression"].isin(group_webp + group_neutral)]
    df_jpeg = df_ds_model[df_ds_model["compression"].isin(group_jpeg + group_neutral)]

    # ============================================================
    # RATE–ACCURACY
    # ============================================================
    plt.figure(figsize=(10, 7))

    # WebP trend
    if len(df_webp) > 1:
        dfw = df_webp.sort_values("compression_ratio")
        xs, ys = smooth_line(dfw["compression_ratio"], dfw["test_accuracy"])
        plt.plot(xs, ys, color=COLOR_WEBP, linewidth=3, alpha=0.9, label="WebP")

    # JPEG trend
    if len(df_jpeg) > 1:
        dfj = df_jpeg.sort_values("compression_ratio")
        xs, ys = smooth_line(dfj["compression_ratio"], dfj["test_accuracy"])
        plt.plot(xs, ys, color=COLOR_JPEG, linewidth=3, alpha=0.9, label="JPEG")

    # Scatter (legend auto-generated)
    plt.scatter(
        df_ds_model["compression_ratio"],
        df_ds_model["test_accuracy"],
        s=160,
        c=df_ds_model["color"],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        label=None   # suppress default label
    )

    # ---------------------------------------
    # SIMPLE LEGEND (schemes + trends)
    # ---------------------------------------
    scatter_handles = []
    scatter_labels = []

    # generate one legend entry per compression scheme
    for comp in df_ds_model["compression"].unique():
        scatter_handles.append(
            Line2D([0], [0],
                   marker="o",
                   color="white",
                   markerfacecolor=color_map[comp],
                   markeredgecolor="black",
                   markersize=12,
                   label=comp)
        )
        scatter_labels.append(comp)

    # trend handles
    trend_handles = [
        Line2D([0], [0], color=COLOR_WEBP,  linewidth=3, label="WebP"),
        Line2D([0], [0], color=COLOR_JPEG, linewidth=3, label="JPEG")
    ]

    plt.legend(
        handles=scatter_handles + trend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Compression Schemes"
    )

    plt.xlabel("Compression Ratio (bits per pixel)")
    plt.ylabel("Test Accuracy")
    plt.title(f"Rate–Accuracy Curve\n{model_name} on {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"results/rate-acc-{model_name}-{dataset_name}.png", dpi=320)
    plt.close()

    # ============================================================
    # RATE–DISTORTION
    # ============================================================
    plt.figure(figsize=(10, 7))


    if len(df_webp) > 1:
        dfw = df_webp.sort_values("compression_ratio")
        xs, ys = smooth_line(dfw["compression_ratio"], dfw["train_loss"])
        plt.plot(xs, ys, color=COLOR_WEBP, linewidth=3, alpha=0.9, label="WebP Trend")

    if len(df_jpeg) > 1:
        dfj = df_jpeg.sort_values("compression_ratio")
        xs, ys = smooth_line(dfj["compression_ratio"], dfj["train_loss"])
        plt.plot(xs, ys, color=COLOR_JPEG, linewidth=3, alpha=0.9, label="JPEG Trend")

    plt.scatter(
        df_ds_model["compression_ratio"],
        df_ds_model["train_loss"],
        s=160,
        c=df_ds_model["color"],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        label=None
    )

    # legend again
    plt.legend(
        handles=scatter_handles + trend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Compression Schemes"
    )

    plt.xlabel("Compression Ratio (bits per pixel)")
    plt.ylabel("Cross Entropy Loss")
    plt.title(f"Rate–Distortion Curve\n{model_name} on {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"results/rate-dist-{model_name}-{dataset_name}.png", dpi=320)
    plt.close()


# df = pd.read_csv("results/MLP_results.csv")
df = pd.read_csv("results.csv")
metrics = ["test_accuracy", "avg_epoch_time", "train_loss", "test_time"]

# Convert metrics to numeric (CSV stores them as strings)
for col in metrics:
    df[col] = pd.to_numeric(df[col])


# Normalize by baseline (no compression) per dataset
# Subtract baseline accuaracy and training loss
# Divide baseline times
df_improvements = df.copy()

for metric in [metrics[0], metrics[2]]:  # accuracy, train_loss
    df_improvements[metric] = df.groupby(["dataset", "model"])[metric].transform(lambda x: x - x[df.loc[x.index, "compression"] == "none"].iloc[0])

# plot_accuracy_by_dataset(df_improvements)
# plot_training_speed_by_dataset(df)
# plot_inference_speed_by_dataset(df)
# plot_heatmaps_by_dataset(df, "test_accuracy")
# plot_heatmaps_by_dataset(df, "test_time")
# plot_heatmaps_by_dataset(df, "avg_epoch_time")

plot_rate_distortion(df, dataset_name="cifar10", model_name="small-CNN")
