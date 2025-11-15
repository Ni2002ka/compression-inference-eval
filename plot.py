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

plot_accuracy_by_dataset(df_improvements)
# plot_training_speed_by_dataset(df)
# plot_inference_speed_by_dataset(df)
