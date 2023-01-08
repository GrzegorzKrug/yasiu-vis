# Readme of `yasiu.visualisation`

High level functions, to quickly visualise data frames.

## Installation

```shell
pip install yasiu.visualisation
```

## Sequence reader Generators

- `summary_plot` - plot dataframe, possible grouping by columns

### Import:

```py
from yasiu.visualisation import summary_plot
```

### Use example:

```py
summary_plot(df)
summary_plot(df, group="column-name")
summary_plot(df, group="column-name", split_widnow="column")
```