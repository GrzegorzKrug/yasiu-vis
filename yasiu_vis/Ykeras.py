import matplotlib.pyplot as _plt
import numpy as _np
# from matplotlib.cm import get_cmap as _get_cmap
from matplotlib.colors import Normalize as _Normalize


def plotLayersWeights(
    layers, innerCanvas=2, midScale=0.8,
        numFmt=">4.2f",
        figsize=(40, 30), dpi=100,
        drawVertical=False, separateFirstLast=True,
        normalizeValues=False,
):
    """
    Draws layers weights onto matplotlib figure.
    innerCanvas: rows/columns for hidden layers.
    numFmt: number formatter
    drawVertical: stack layers in vertical or horizontal direction
    separateFirstLast: draw first last layer independent from hidden layers
    normalizeValues: Use normalization for color map per ax <min ,max>
    """
    if not isinstance(layers, (list,)):
        layers = [layers]

    canvasSizes = []
    if len(layers) > 2:
        if separateFirstLast:
            shapes = [lay.get_weights()[0].shape for lay in layers[1:-1]]
        else:
            shapes = [lay.get_weights()[0].shape for lay in layers]
        # print("Shapes", shapes, len(shapes))
        htemp = []
        for i in range(len(shapes)):
            if i % innerCanvas and innerCanvas > 1:
                "Skip columnes other ahtn first"
                continue

            if (i+innerCanvas) < len(shapes):
                if innerCanvas <= 1:
                    shape_scope = [shapes[i]]
                else:
                    shape_scope = shapes[i: i+innerCanvas-1]
                sizes = [min(sh) for sh in shape_scope]
                htemp.append(max(sizes) * midScale)

            else:
                # h1, w1 = shapes[i]
                htemp.append(min(shapes[i]) * midScale)
        canvasSizes = htemp
        del htemp

    if separateFirstLast:
        canvasSizes = [
            1.0,
            *canvasSizes,
            1.0
        ]

    canvasSizes = [c if c > 2.0 else 2.0 for c in canvasSizes]
    all_axes = []

    # print("All ratios:", canvasSizes)
    plots_num = len(canvasSizes)

    fig = _plt.figure(figsize=figsize, dpi=dpi)
    if drawVertical:
        grid = fig.add_gridspec(len(canvasSizes), 1, height_ratios=canvasSizes)
    else:
        grid = fig.add_gridspec(1, len(canvasSizes),  width_ratios=canvasSizes)

    if separateFirstLast:
        "Create separate input"
        ax_first = fig.add_subplot(grid[0, 0])
        all_axes.append(ax_first)

    if separateFirstLast:
        "Skip first row/col"
        end = plots_num - 2 + 1
        start = 1
    else:
        end = plots_num
        start = 0

    for i in range(start, end):
        for j in range(innerCanvas):
            if drawVertical:
                ax = fig.add_subplot(
                    grid[i, 0].subgridspec(1, innerCanvas)[0, j])
            else:
                ax = fig.add_subplot(
                    grid[0, i].subgridspec(innerCanvas, 1)[j, 0])
            all_axes.append(ax)
            # print(f"axes now: {len(all_axes)}")

    if separateFirstLast:
        if drawVertical:
            ax_last = fig.add_subplot(grid[-1, 0])
        else:
            ax_last = fig.add_subplot(grid[0, -1])
        all_axes.append(ax_last)

    # print(f"All axes: {len(all_axes)}")
    # print(f"All Layer: {len(layers)}")
    if normalizeValues:
        "Allow matplotlib to normalize values"
        my_norm = None
    else:
        my_norm = _Normalize(vmin=-1, vmax=1)

    for lind, lay in enumerate(layers):
        # print(f"Plotting layer index {lind} ({lay.name})")
        ax = all_axes[lind]

        weights, biases = lay.get_weights()
        weights_visualization = weights.copy()
        biases_visualization = _np.expand_dims(
            biases, axis=0)  # Rozszerzenie wymiaru biasów

        # Połączenie wag i biasów w jedną tablicę dla wizualizacji
        combined_visualization = _np.vstack(
            [weights_visualization, biases_visualization])

        is_bias_onRight = False
        H, W = combined_visualization.shape

        if lind == 0:
            # is_bias_onRight = True
            if (drawVertical and (H > W)) or not drawVertical and W > H:
                combined_visualization = combined_visualization.T
                is_bias_onRight = True

        elif lind == len(layers)-1:
            ax = all_axes[-1]
            if drawVertical:
                combined_visualization = combined_visualization.T
                is_bias_onRight = True

        else:
            if (H > W) and drawVertical or not drawVertical and W > H:
                combined_visualization = combined_visualization.T
                is_bias_onRight = True

        ax.matshow(combined_visualization, cmap="viridis", norm=my_norm)

        for (i, j), val in _np.ndenumerate(combined_visualization):
            ax.text(j, i, f"{val:{numFmt}}", ha='center',
                    va='center', color='white', fontsize=10)

        if is_bias_onRight:
            ticks = ax.get_xticks()[1:-1]
            tick_strings = [f"W_{int(val)}" for val in ticks]
            tick_strings[-1] = "Bias"
            ax.set_xticks(ticks, tick_strings)
            ax.set_ylabel("Nodes")

        else:
            ticks = ax.get_yticks()[1:-1]
            tick_strings = [f"W_{int(val)}" for val in ticks]
            tick_strings[-1] = "Bias"
            ax.set_yticks(ticks, tick_strings)
            ax.set_xlabel("Nodes")

        ax.set_title(f" Layer: {lind} ({lay.name})")
        ax.grid(0)

    _plt.tight_layout()
    if drawVertical:
        _plt.subplots_adjust(hspace=0.1, wspace=0.01,)
    else:
        _plt.subplots_adjust(wspace=0.03, hspace=0.07,)


__all__ = [
    "plotLayersWeights"
]

if __name__ == "__main__":
    inputSize = 2

    import yasiu_vis.Ykeras as Ykeras

    model = Ykeras.models.Sequential()
    model.add(Ykeras.Input(shape=(inputSize,)))
    model.add(Ykeras.layers.Dense(30, activation="leaky_relu"))
    model.add(Ykeras.layers.Dense(30, activation="leaky_relu"))
    model.add(Ykeras.layers.Dense(30, activation="leaky_relu"))
    model.add(Ykeras.layers.Dense(21, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    model.add(Ykeras.layers.Dense(10, activation="leaky_relu"))
    model.add(Ykeras.layers.Dense(1, activation="linear"))

    optim = Ykeras.optimizers.Adam(learning_rate=0.0004)
    model.compile(loss="mse", optimizer=optim)

    plotLayersWeights(model.layers, innerCanvas=2, figsize=(42, 25), dpi=70)
    _plt.savefig("temp.png")
