import pytest
import pandas as pd
import matplotlib.pyplot as plt

from easy_draw import summary_plot


@pytest.fixture()
def short_data_frame():
    df = pd.DataFrame(columns=['colA', 'colB', 'colC'], dtype=float)
    for i in range(15):
        # df.loc[i, :] = [f'{i}A', f'{i}B', f'{i}C']
        df.loc[i, :] = [i, i, i]
    # df.loc[1, :] = ['A', 'B', 'C']
    # df.loc[2, :] = ['A', 'B', 'C']
    # print()
    # print(df)
    # print(df.dtypes)

    return df


# @pytest.fixture()
def test_first(short_data_frame):
    summary_plot


def test_first(short_data_frame):
    # pass
    summary_plot(short_data_frame, show=False)
    # import matplotlib
    # pass
    plt.close()
