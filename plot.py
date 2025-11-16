import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




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
    plt.show()


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
    plt.show()



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
            cbar_kws={"label": "Accuracy"}
        )

        plt.title(f"{metric} Heatmap — {ds}", fontsize=16)
        plt.xlabel("Model")
        plt.ylabel("Compression")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()



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
plot_heatmaps_by_dataset(df, "test_accuracy")
plot_heatmaps_by_dataset(df, "test_time")
plot_heatmaps_by_dataset(df, "avg_epoch_time")
