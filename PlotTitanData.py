import matplotlib.pyplot as plt
import pandas as pd
import os

from yasiu_vis.ypandas import summary_plot
from matplotlib.style import use

if __name__ == "__main__":
    use("ggplot")

    data = pd.read_csv(os.path.join("tests", "data", "titanic_train.csv"))
    summary_plot(
        data,
        group_key="Destination",
        desired_rows=4, max_groups=10, logy=True,
        figure_params=dict(figsize=(16, 10), dpi=120),
        # split_windows="group"
    )

    plt.savefig(os.path.join("pics", "SpaceTitan.png"))
    # plt.show()
