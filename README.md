# Readme of `yasiu-vis`

High level functions, to quickly visualise data frames.

## Installation

```shell
pip install yasiu-vis
```

## Sequence reader Generators

- `summary_plot` - plot dataframe, possible grouping by columns

### Use example:

```py
from yasiu_vis.pandas import summary_plot


# df: pandas.DataFrame

summary_plot(df)
summary_plot(df, group="column-name")
summary_plot(df, group="column-name", split_widnow="column")
```

```py
from yasiu_vis.keras import plotLayersWeights
from matpotlib import pyplot as plt
import keras

model : keras.models.Sequential
plotLayersWeights(model.layers)
plt.show()
```
# All packages

[1. Native Package](https://pypi.org/project/yasiu-native/)

[2. Math Package](https://pypi.org/project/yasiu-math/)

[3. Image Package](https://pypi.org/project/yasiu-image/)

[4. Visualisation Package](https://pypi.org/project/yasiu-vis/)

