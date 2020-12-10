import math
import json
import datetime
import numbers
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']

col_names = {
    "infected_a": "Infected - Mild",
    "infected_b": "Infected - Hospitalized",
    "infected_c": "Infected - ICU",
    "dead": "Dead",
    "asymp": "Asymptomatic",
    "exposed": "Exposed",
}

col_colors = {
    "Infected - Mild": "xkcd:light blue",
    "Infected - Hospitalized": "xkcd:green",
    "Infected - ICU": "xkcd:magenta",
    "Dead": "xkcd:red",
    "Asymptomatic": "xkcd:teal",
    "Exposed": "xkcd:purple",
}


def plot_df(
    df_to_plot, cols, line_day=None, interventions=None, title="", y_max=8000000, filename=None
):
    cols.append("date")

    r_effective_flag = False

    if "R effective" in cols:
        cols.remove("R effective")
        r_effective_flag = True

        r_effective_df = df_to_plot.loc[:, ["date", "R effective"]]
        min_date = r_effective_df["date"].min()
        max_date = r_effective_df["date"].max()

    df_to_plot = df_to_plot.loc[:, cols]
    x_dates = df_to_plot["date"].dt.strftime("%Y-%m-%d").sort_values().unique()
    df_to_plot.set_index("date", inplace=True)

    df_to_plot.columns = [col_names[col] for col in df_to_plot.columns]
    stacked = df_to_plot.stack().reset_index()
    stacked.columns = ["date", "Population", "Number of people"]

    # use the col names to set colors for the palette so they stay the same
    colors = {col: col_colors[col] for col in df_to_plot.columns if col != "date"}

    # make the population range into the max + 10%
    y_max = stacked["Number of people"].max() * 1.1

    if line_day is None:
        line_day = datetime.datetime.now() - datetime.timedelta(days=2)

    if r_effective_flag is True:
        gridkw = dict(height_ratios=[1, 5])
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw=gridkw, figsize=(16, 14))

        ax2.axvline(line_day, 0, y_max, linestyle="--", color="darkblue")
        trans = ax2.get_xaxis_transform()

        plt.text(
            line_day + datetime.timedelta(days=2), 0.95, "latest data", transform=trans,
        )

        r_eff_max = math.ceil(r_effective_df['R effective'].max()) + 1

        ax2.set_ylim([0, y_max])

        ax1.set_ylim([0, r_eff_max])
        ax1.hlines(1, min_date, max_date, linestyles="dashed")

        sb.lineplot(x="date", y="R effective", data=r_effective_df, ax=ax1)
        sb.lineplot(
            x="date",
            y="Number of people",
            hue="Population",
            palette=colors,
            data=stacked,
            ax=ax2,
        )
    else:
        plt.figure(figsize=(15, 8))
        plt.ylim([0, y_max])

        sb.lineplot(
            x="date",
            y="Number of people",
            hue="Population",
            palette=colors,
            data=stacked,
        )
        #plt.axvline(line_day, 0, y_max, linestyle="--", color="darkblue")
        #plt.text(
        #    line_day + datetime.timedelta(days=2), 0.95 * y_max, "latest data",
        #)


    label_height = [0.9, 0.875, 0.85, 0.825, 0.8, 0.775, 0.75, 0.725, 0.75]

    print(f'interventions: {interventions}')

    if interventions is not None:
        line_list = []
        for i, intervention in enumerate(interventions):
            if r_effective_flag is True:
                ax2.axvline(
                    intervention.intervention_start_date,
                    0,
                    y_max,
                    color="dimgrey",
                    linestyle="--",
                )
                plt.text(
                    intervention.intervention_start_date + datetime.timedelta(days=2),
                    label_height[i],
                    intervention.name,
                    transform=trans,
                )
            else:
                plt.axvline(
                    intervention.intervention_start_date,
                    0,
                    y_max,
                    color="dimgrey",
                    linestyle="--",
                )
                plt.text(
                    intervention.intervention_start_date + datetime.timedelta(days=2),
                    label_height[i] * y_max,
                    intervention.name,
                )

    plt.title(title)

    if filename is not None:
        plt.savefig(filename)

    return plt


def prep_plot(
    prep_df, chart_cols, line_day=None, interventions=None, title="", y_max=8000000, filename=None
):
    prep_df.loc[:, "date"] = pd.to_datetime(prep_df["date"])

    first_case_date = prep_df.loc[(prep_df.infected > 0), "date"].min()
    peak_date = prep_df.loc[(prep_df.infected_b == prep_df.infected_b.max())][
        "date"
    ].values[0]
    peak = prep_df.loc[(prep_df.infected_b == prep_df.infected_b.max())][
        "infected_b"
    ].values[0]

    icu_peak_date = prep_df.loc[(prep_df.infected_c == prep_df.infected_c.max())][
        "date"
    ].values[0]
    icu_peak = prep_df.loc[(prep_df.infected_c == prep_df.infected_c.max())][
        "infected_c"
    ].values[0]

    deaths = prep_df.loc[:, "dead"].max()

    print("first case")
    print(first_case_date)
    print("peak in hospitalizations")
    print(peak_date)
    print(f"{peak:,}")
    print("peak in icu")
    print(icu_peak_date)
    print(f"{icu_peak:,}")
    print("deaths")
    print(f"{deaths:,}")

    plot_df(
        prep_df,
        chart_cols,
        line_day,
        interventions,
        f"{title}. Peak hospitalizations: {int(peak):,}. Deaths: {int(deaths):,}",
        y_max,
        filename,
    )


def plot_actuals(model_df, actuals_df, model_cols, title, y_max=8000000):

    combo_df = pd.merge(model_df, actuals_df, how="outer", on="date")

    plot_df(
        combo_df, model_cols, f"{title}.", y_max,
    )


def report_months(df):
    date_list = [
        datetime.datetime(2020, 5, 1).date(),
        datetime.datetime(2020, 6, 1).date(),
        datetime.datetime(2020, 7, 1).date(),
        datetime.datetime(2020, 8, 1).date(),
        datetime.datetime(2020, 9, 1).date(),
        datetime.datetime(2020, 10, 1).date(),
        datetime.datetime(2020, 11, 1).date(),
        datetime.datetime(2020, 12, 1).date(),
    ]

    cols = {
        "date": "Date",
        "infected_a": "Infected",
        "infected_b": "Hospitalized",
        "infected_c": "ICU",
        "dead": "Deaths",
    }

    report_df = df.loc[(df["date"].isin(date_list)), list(cols.keys())]

    report_df.rename(columns=cols, inplace=True)

    return report_df.T
