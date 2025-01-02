import pytest

import os

from yasiu_vis.ykeras import plotLayersWeights
from matplotlib import pyplot as plt

try:
    import tensorflow
except Exception as err:
    os.environ["KERAS_BACKEND"] = "torch"
    # import keras

import keras


def new_model():
    inputSize = 4
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(inputSize,)))
    model.add(keras.layers.Dense(20, activation="leaky_relu"))
    model.add(keras.layers.Dense(10, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    # model.add(keras.layers.Dense(10, activation="leaky_relu"))
    model.add(keras.layers.Dense(2, activation="linear"))

    optim = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="mse", optimizer=optim)

    return model


TEST_MODEL = new_model()


def test_plotSequential1():
    model = new_model()
    plotLayersWeights(model.layers)
    plt.close()


def test_plotSequential2():
    model = new_model()
    plotLayersWeights(model.layers)
    plt.close()


def test_plotSequential3():
    model = new_model()
    plotLayersWeights([model.layers[0]])
    plt.close()


def test_plotSequential4():
    model = new_model()
    plotLayersWeights(model)
    plt.close()


@pytest.mark.parametrize("arg", [TEST_MODEL, TEST_MODEL.layers])
@pytest.mark.parametrize("innerCanvas", [None])
@pytest.mark.parametrize("midScale", [None])
@pytest.mark.parametrize("drawVertical", [None])
@pytest.mark.parametrize("separateFirstLast", [None])
@pytest.mark.parametrize("normalizeColors", [None])
@pytest.mark.parametrize("scaleWeights", [None])
def test_Nones(
    arg,
    innerCanvas, midScale,
    drawVertical, separateFirstLast, normalizeColors,
    scaleWeights
):
    # model = new_model()
    plotLayersWeights(
        arg,
        innerCanvas=innerCanvas,
        midScale=midScale,
        drawVertical=drawVertical,
        # figsize=(3, 4), dpi=40,
        figsize=(10, 5),
        dpi=20,
        separateFirstLast=separateFirstLast,
        normalizeColors=normalizeColors,
        scaleWeights=scaleWeights,
    )
    plt.close("all")


@pytest.mark.parametrize("arg", [TEST_MODEL, TEST_MODEL.layers])
@pytest.mark.parametrize("innerCanvas", [1, 3])
@pytest.mark.parametrize("midScale", [0, 0.9])
@pytest.mark.parametrize("drawVertical", [True, False])
@pytest.mark.parametrize("separateFirstLast", [True, False])
@pytest.mark.parametrize("normalizeColors", [True, False])
@pytest.mark.parametrize("scaleWeights", [10.0, 100])
def test_arguments(
    arg,
    innerCanvas, midScale,
    drawVertical, separateFirstLast, normalizeColors,
    scaleWeights
):
    # model = new_model()
    plotLayersWeights(
        arg,
        innerCanvas=innerCanvas,
        midScale=midScale,
        drawVertical=drawVertical,
        # figsize=(3, 4), dpi=40,
        figsize=(10, 5),
        dpi=20,
        separateFirstLast=separateFirstLast,
        normalizeColors=normalizeColors,
        scaleWeights=scaleWeights,
    )
    plt.close("all")
