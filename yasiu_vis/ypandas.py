import matplotlib as _mpl
import numpy as _np

import pandas as _pd

from matplotlib import pyplot as _plt
from matplotlib.lines import Line2D as _Line2D

from yasiu_math.math import round_number as _round_number

import warnings as _warn


def get_grid_dims(size, desiredRows=None):
    if size <= 0:
        return 0, 0

    if (desiredRows):
        rows = desiredRows
        cols = _np.ceil(size / rows)
    else:
        sq = _np.sqrt(size)
        rows = _np.floor(sq)
        cols = rows + 1

    if (rows * cols) < size:
        rows = cols

    while ((cols - 1) * rows) >= size:
        cols -= 1
        # print(cols)

    return int(rows), int(cols)


def summary_plot(
    data_df: _pd.DataFrame, group_key: str = None, max_groups: int = 9,
    figure_params: dict = None, plot_params: dict = None, drawGrid=True,
    logy: bool = False, logx: bool = False,
    split_windows: str = 'None',
    legend_place: str = 'subplot',
    desired_rows=None,
    show: bool = False,
):
    """
    Creates new figure and plots data.

    Args:
        `data_df`: `pandas.DataFrame`
            Input data frame

        `group_key`: `str` (optional), defaults to None.
            Plot distribution from subgroups by groupping values in given category.
            Must match exact value in df.column. Use `max_groups`

        `max_groups`: `int` (optional), defaults to 9.
            Maximal amount of subgroups. Used for spliting continuous values int bins.

        `figure_params`: `dict` (optional), defaults to None.
            dict used to affect figure creation.

        `plot_params`: `_type_` (optional), defaults to None.
            Dict of keyword arguments passed to matplotlib.pyplot.hist()
            Example:
            `plot_params=dict(bins=10)`

        `drawGrid`: `bool` (optional), defaults to True.
            Flag to draw grid on each graph.

        `logy`: `bool` (optional), defaults to False.
            Y axis log scaling

        `logx`: `bool` (optional), defaults to False.
            X axis log scaling

        `split_windows`: `str` (optional), defaults to 'None'.
            group - new window for each category

            column - new window for each column

            category - same as column

        `legend_place`: `str` (optional), defaults to 'subplot'.
            Position of the legend

            'subplot' - legend is created in extra blank subplot

            'new' - same as `subplot`

            'same' - last subplot

            'external' - legend is placed in separate window

        `desired_rows` : `int` (optional), default to None
            Sets exact amount of rows in subplot layout.

        `show`: `bool` (optional), defaults to False.
            Flag meant to call `matplotlib.pyplot.show()`
    """
    data_df = data_df.copy()
    total_columns = data_df.shape[1]
    def_hspace = 0.4

    "Check or initialize passed arguments"
    if figure_params is None:
        figure_params = dict()
    else:
        assert isinstance(
            figure_params,
            dict), f"Figure params must be dict type! but got: {type(figure_params)}"

    if 'figsize' not in figure_params:
        figure_params['figsize'] = 10, 7

    if 'dpi' not in figure_params:
        figure_params['dpi'] = 100

    if plot_params is None:
        plot_params = dict()
    else:
        assert isinstance(
            plot_params,
            dict), f"Plot params must be dict type! but got: {type(plot_params)}"

    if 'alpha' not in plot_params:
        plot_params['alpha'] = 0.6

    if group_key is not None:
        max_groups = int(max_groups)
        if max_groups <= 1:
            _warn.warn("max_groups must be higher than 1. Omiting `group_key`")
            group_key = None

    """
    SPLIT
        Split by group
        Split by category/column
    """
    split_windows = split_windows.lower()
    if split_windows in ['column', 'category']:
        if group_key is not None:
            figure_list = [_plt.figure(**figure_params)
                           for _ in range(total_columns - 1)]
        else:
            figure_list = [_plt.figure(**figure_params)
                           for _ in range(total_columns)]
    elif split_windows in ['group']:
        figure_list = []

    else:
        _plt.figure(**figure_params)
        figure_list = None

    if group_key is not None:

        uniq = data_df[group_key].unique()
        if (uniq.shape[0] > max_groups and not _pd.api.types.is_numeric_dtype(data_df[group_key])):
            print(
                f"Selected column `{group_key}` is not numeric and `max_groups` is smaller than unique count: {uniq.shape[0]}.")
            return

        "Group data"
        group_dict = get_dataframe_groups(
            data_df, group_key=group_key, max_groups=max_groups)

        if legend_place == 'same':
            total_columns -= 1

        if split_windows in ['column', 'category']:
            plot_rows, plot_cols = get_grid_dims(len(group_dict), desiredRows=desired_rows)
        else:
            plot_rows, plot_cols = get_grid_dims(total_columns, desiredRows=desired_rows)

        for ind, (group_name, value) in enumerate(group_dict.items()):
            "Plot given selection matching to criteria"

            plot_params['label'] = group_name

            if split_windows == "group":
                fig = _plt.figure(**figure_params)
                figure_list.append(fig)
                curFigList = None
                title = None

            elif split_windows in ['column', 'category']:
                title = group_name
                curFigList = figure_list
            else:
                title = None
                curFigList = None

            iterate_split_plot(
                value, plot_rows, plot_cols,
                grid=drawGrid, plot_params=plot_params,
                logy=logy, logx=logx,
                figure_list=curFigList,
                subplot_ind=ind + 1,
                title=title,
            )

            if split_windows in ['column', 'category']:
                pass
                # _create_legend(list(group_dict.keys()))
                # plt.suptitle(f"{key}")

            elif split_windows == 'group':
                # _create_legend(list(group_dict.keys()))
                _plt.suptitle(f"Values in group '{group_name}'")

            else:
                "No slip, stacked images"
                "Put legend as group"
                _plt.suptitle(f"Groupping by '{group_key}'")

        if split_windows not in ['column', 'category', 'group']:
            _plt.tight_layout()
            if legend_place == 'subplot' or legend_place == 'new':
                _plt.subplot(plot_rows, plot_cols, total_columns)
                _create_legend(list(group_dict.keys()))
                _plt.axis('off')

            elif legend_place == 'same':
                _plt.subplot(plot_rows, plot_cols, total_columns)
                _create_legend(list(group_dict.keys()))

            elif legend_place == 'external':
                _plt.figure()
                _create_legend(list(group_dict.keys()))
                _plt.axis('off')
            else:
                raise KeyError(f"Unsupported legend place: {legend_place}")
            # plt.tight_layout()

        if figure_list:
            for ind, figure in enumerate(figure_list):
                columns = data_df.columns
                _plt.figure(figure.number)
                _plt.tight_layout()
                if split_windows in ['column', 'category']:
                    _plt.suptitle(f"Values of '{columns[ind]}' per groupping")

    else:
        if split_windows in ['column', 'category']:
            # figure_list = [plt.figure(**figure_params) for _ in range(total_columns)]
            plot_rows, plot_cols = 1, 1
            subplot_ind = 1
        elif split_windows == 'group':
            raise KeyError(
                "Windows splitting: `group` not available without groupping")

        else:
            plot_rows, plot_cols = get_grid_dims(total_columns, desiredRows=desired_rows)
            # figure_list = None
            subplot_ind = None

        iterate_split_plot(
            data_df, plot_rows, plot_cols, drawGrid, plot_params,
            logy=logy, logx=logx, figure_list=figure_list,
            subplot_ind=subplot_ind,
        )
        # plt.subplots_adjust(hspace=def_hspace)
        _plt.tight_layout()

    if figure_list:
        for figi in figure_list:
            _plt.figure(figi.number)
            _plt.subplots_adjust(hspace=def_hspace)
            _plt.tight_layout()

    _plt.tight_layout()

    if show:
        _plt.show()


def _create_legend(names_list, size=10):
    color_cycler = _mpl.rcParams['axes.prop_cycle']

    actors = []
    for name, style_dict in zip(names_list, color_cycler):
        line = _Line2D([0, 0], [0, 0], linewidth=5, **style_dict)
        actors.append(line)

    _plt.legend(actors, names_list, loc='upper right')
    # plt.axis('off')


def iterate_split_plot(
        data_df, plot_rows, plot_cols, grid, plot_params, logx=False, logy=False,
        figure_list=None, subplot_ind=None, title=None, suptitle=None, rotation=15,
):
    """
    Plot every column separately

    Args:
        data_df:
        plot_rows:
        plot_cols:
        grid:
        plot_params:
        logx:
        logy:
        figure_list:
        subplot_ind:
        title:

    Returns:

    """
    if figure_list is not None:
        assert len(
            figure_list) == data_df.shape[1], f"{len(figure_list)}, {data_df.shape[1]}"

    for col_ind, (name, value) in enumerate(data_df.items()):
        if figure_list is not None:
            _plt.figure(figure_list[col_ind].number)
            if subplot_ind:
                _plt.subplot(plot_rows, plot_cols, subplot_ind)
        else:
            _plt.subplot(plot_rows, plot_cols, col_ind + 1)

        if title:
            _plt.title(title)
        else:
            _plt.title(name)

        "Cast values for _plt.hist"
        # print(f"col index: {col_ind}, name: {name}, type: {value.dtype}")
        if isinstance(value.dtype, _pd.CategoricalDtype):
            value = value.astype("string")
        elif _pd.api.types.is_object_dtype(value.dtype):
            "UNDEFINED TYPE, TRY FLOAT OR STRING "
            convertSuccess = False

            try:
                value = value.astype(float)
                # print("CONVERTED TO FLOAT!")
                convertSuccess = True
            except Exception:
                pass

            try:
                if (not convertSuccess):
                    value = value.astype("string")
                    convertSuccess = True
                    # print(f"CONVERTED TO STR! {value.dtype}")
            except Exception as err:
                # print(f"Can't conver to str: {err}")
                pass

            # print("RETURNING!")
            # continue
            if (not convertSuccess):
                print(
                    f"Not suppoted column found in dataframe: {name}, can not convert to float or string.")
                continue

        # print(f"col index: {col_ind}, name: {name}, type: {value.dtype}")
        nanMask = value.isna()
        value = value[~nanMask]

        # print(f"Plotting: {value.dtype}, {value.shape}, params: {plot_params}")
        if isinstance(value.dtype, (_pd.StringDtype, bool)) or value.dtype == bool or value.dtype == _np.bool:
            # print(type(value))
            # value : _pd.Series
            ct = value.value_counts()
            # print(f"Count: {ct}")
            if (ct.shape[0] < 10):
                _plt.bar(ct.index, ct, log=logy, **plot_params)
            else:
                # print(f"Too many items in {name}! Plotting unique value count")
                _plt.xlabel("Shared value")
                _plt.hist(ct, log=logy, **plot_params)

            loc, label = _plt.xticks()
            # print(loc)
            # print(lab)
            # clean_numbers = [int(l.get_text().replace('−', '-')) for l in lab]
            if (ct.shape[0] < 10):
                pass
                # print(ct.index)
                # print(label)
                # _plt.xticks(loc, ct.index.astype(str), rotation=30)
            else:
                _plt.xticks(loc, _np.array(loc, dtype=int), rotation=30)

        else:
            _plt.hist(value, log=logy, **plot_params)

        _plt.xticks(rotation=rotation)

        if suptitle is not None:
            _plt.suptitle(suptitle)

        if grid:
            _plt.grid(True)

        if logy and logx:
            _plt.loglog()
        elif logy:
            _plt.semilogy()
        elif logx:
            _plt.semilogx()


def get_dataframe_groups(data_df, group_key, max_groups=10):
    filter_column = data_df.pop(group_key)
    filter_array = _np.array(filter_column)
    output_dict = dict()

    unique_vals = filter_column.unique()

    grouped_keys_dict = cluster_keys(unique_vals, max_groups=max_groups)
    for key_name, group_list_values in grouped_keys_dict.items():
        desired_keys = _np.array(group_list_values).reshape(-1, 1)

        mask = (desired_keys == filter_array).any(axis=0)
        if (_pd.isna(group_list_values).any()):
            nanMask = _pd.isna(filter_array)
            mask = mask | nanMask
            # print(key_name, type(group_list_values))

        minidf = data_df.loc[mask, :]
        output_dict[f"{group_key}={key_name}"] = minidf

    return output_dict


def cluster_keys(keys, max_groups=10, numeric_keys_rounding=5):
    "Find values. Returns dict, in case numeric types must grouped into ranges."
    keys = list(keys)
    output_dict = dict()

    # keys = keys[:5]

    "Check if all values are numeric"
    are_allNumeric = True
    # print(f"Clustering keys: {keys}")
    for key in keys:
        if not isinstance(key, (int, float, _np.number)) or type(key) is bool or type(key) is _np.bool:
            are_allNumeric = False
            break

    if len(keys) > max_groups:
        keys.sort()

        "<ADD NUMERIC CLUSTERING HERE>"
        # size = len(keys)
        # step = size / max_groups
        # step = np.floor(step).astype(int)

        output_dict = dict()
        # last_key = None

        "FIRST"
        "Linspace"
        indexes = _np.linspace(
            0, len(keys), max_groups + 1).round().astype(int)
        # print("indexes")
        # print(indexes)
        # print(f"all keys: {len(keys)}")

        for first, last in zip(indexes, indexes[1:]):
            select = keys[first:last]
            if are_allNumeric:
                short_nums = [str(_round_number(num)) for num in select]
                key = shrink_array_to_string(short_nums, )
            else:
                key = shrink_array_to_string(select, )
            key = shrink_array_to_string(select, )

            output_dict[key] = select

    else:
        for k in keys:
            # print(f"All are numeric: {are_allNumeric}")
            if are_allNumeric:
                this_num_str = str(_round_number(k, round=numeric_keys_rounding))
                output_dict[this_num_str] = [k]
            else:
                output_dict[k] = [k]

    return output_dict


def shrink_array_to_string(arr, max_size=20, rounding=5, ignore_minimal_size_error=False):
    if len(arr) <= 2:
        return str(arr)

    max_size -= 2  # for parenthesis []
    max_size -= 5  # for middle `, .. `
    first = arr[0]
    last = arr[-1]

    if isinstance(first, (int, float, _np.number)):
        first = str(_round_number(first))
    if isinstance(last, (int, float, _np.number)):
        last = str(_round_number(last))

    last = "," + last

    size_left = max_size - len(first) - len(last)
    if size_left < 0 and not ignore_minimal_size_error:
        raise ValueError(
            f"Two edge values will exceed minimal required size: `{arr}`, "
            f"First:`{first}`, last:`{last}`, "
            f"Two words size: {len(first) + len(last)}, but got space: {max_size}")

    middle = ""

    left_over = arr[1:-1]

    for ci, next_val in enumerate(left_over):
        if size_left <= 0:
            break

        if isinstance(next_val, (int, float, _np.number)):
            next_val = str(_round_number(next_val, rounding))

        if len(next_val) + 1 <= size_left:
            middle += f",{next_val}"
            size_left -= len(next_val) + 1

        else:
            break

    # print(f"Middle: {middle}")
    # print(f"last val: {next_val}")
    if ci != len(left_over) - 1:
        middle += ", .. "

    return "[" + first + middle + last + "]"


__all__ = ["shrink_array_to_string", "summary_plot",
           "get_grid_dims", "iterate_split_plot"]


def random_data_frame(rows_n=100, columns_N=10, classes_N=5):
    columns = [f"col-{chr(num + 65)}" for num in range(columns_N)]
    columns += ['class']
    # print("columns:")
    # print(columns)

    df = _pd.DataFrame(columns=columns)
    for ind in range(rows_n):

        "Random Numbers"
        rnd = _np.random.random(columns_N + 1)
        df.loc[ind] = rnd

        "Random Classes"
        cls = _np.random.randint(0, classes_N)
        df.iloc[ind, columns_N] = cls

    return df


if __name__ == "__main__":
    import os
    _np.arange(5)
    import numpy as np

    # df = random_data_frame(150, columns_N=10, classes_N=8)
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = _pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    # print(df)
    print(df.head)
    summary_plot(
        df,
        # group_key='class',
        group_key='petal width (cm)',
        # split_windows='column',
        # split_windows='group',
        max_groups=4,
        plot_params=dict(alpha=0.7),
        legend_place='subplot'
    )
    _plt.suptitle("Iris dataset. Groupping by 'petal width'. 4 Groups")
    _plt.savefig(os.path.join(os.path.dirname(__file__),
                              "..", "pics", "summaryPlot.png"))
    _plt.show()
