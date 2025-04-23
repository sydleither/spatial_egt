import matplotlib.pyplot as plt
import seaborn as sns


def label_statistic(features, split_char=" "):
    extra = [
        "Sensitive", "Resistant",
        "Local", "Global",
        "Mean", "SD", "Skew", "Kurtosis",
        "Min", "Max", "0"
    ]
    feature_categories = []
    feature_to_statistic = dict()
    for feature in features:
        feature_category = [x for x in feature.split(split_char) if x not in extra]
        feature_category = split_char.join(feature_category)
        feature_to_statistic[feature] = feature_category
        if feature_category not in feature_categories:
            feature_categories.append(feature_category)
    return feature_to_statistic


def format_df(df):
    df["Feature"] = df["Feature"].str.replace("_", " ")
    df["Statistic"] = df["Feature"].map(label_statistic(df["Feature"].unique()))
    return df


def plot_feature_selection(save_loc, measurement, condition, df):
    statistics = df["Statistic"].unique()

    if condition is None:
        file_name = f"{measurement}"
        title = f"Feature {measurement}"
    else:
        file_name = f"{measurement}_{condition}"
        title = f"Feature {measurement}\n{condition}"

    fig, ax = plt.subplots(figsize=(6, len(df["Feature"].unique())//2))
    sns.barplot(
        data=df, x=measurement, y="Feature", ax=ax,
        palette=sns.color_palette("hls", len(statistics)),
        hue="Statistic", hue_order=sorted(statistics)
    )
    ax.set(title=title)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")
    plt.close()
