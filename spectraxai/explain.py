from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataset import Dataset


def bar_plot_importance(
    importance: List, x_labels: List = [], ax: plt.Axes = None
) -> plt.Axes:
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if not x_labels:
        x_labels = ["X{0}".format(i) for i in range(len(importance))]
    pos = np.arange(len(importance))
    ax.bar(pos, importance)
    ax.set_xticks(pos, x_labels)
    return ax


def circular_bar_plot_importance(importance: List, x_labels: List = []) -> plt.Axes:
    if not x_labels:
        x_labels = ["X{0}".format(i) for i in range(len(importance))]

    # Build a dataset
    df = pd.DataFrame({"Name": x_labels, "Value": importance})

    # Reorder the dataframe
    df = df.sort_values(by=["Value"])

    # initialize the figure
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111, polar=True)
    plt.axis("off")

    # Constants = parameters controling the plot layout:
    lowerLimit = 30
    labelPadding = 4

    # Compute max and min in the dataset
    max = df["Value"].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * df.Value + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2 * np.pi / len(df.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index) + 1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles,
        height=heights,
        width=width,
        bottom=lowerLimit,
        linewidth=2,
        edgecolor="white",
        color="#61a4b2",
    )

    # Add labels
    for bar, angle, height, label in zip(bars, angles, heights, df["Name"]):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle,
            y=lowerLimit + bar.get_height() + labelPadding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
        )

    return ax


def correlogram(dataset: Dataset):
    pass
