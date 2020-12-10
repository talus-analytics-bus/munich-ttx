import logging
import math
import json
import datetime
import numbers
from collections import OrderedDict

import numpy as np
import pandas as pd
import boto3

from epi_models.TalusSEIRClass import TalusSEIR as EpiRun, InterventionRun
from ModelClasses import ModelTypes, EpiParams,  Intervention

_logger = logging.getLogger(__name__)

s3 = boto3.resource("s3")

with open('population_largest_cities.json') as f:
    population = json.load(f)


# pulled from run.py, adjusts data to back out dead and recovered
def get_backfill_historical_estimates(df):

    CONFIRMED_HOSPITALIZED_RATIO = 4
    RECOVERY_SHIFT = 13
    HOSPITALIZATION_RATIO = 0.073

    df["estimated_recovered"] = df.cases.shift(RECOVERY_SHIFT).fillna(0) - df.deaths.shift(RECOVERY_SHIFT).fillna(0)
    df["active"] = df.cases - (df.deaths + df.estimated_recovered)
    df["estimated_hospitalized"] = df["active"] / CONFIRMED_HOSPITALIZED_RATIO
    df["estimated_infected"] = df["estimated_hospitalized"] / HOSPITALIZATION_RATIO
    return df


### TODO: need to tie together all the interventions and actuals and get a final
# results set
class ModelRun:
    def __init__(self, state, run_type="city", country="USA", county=None, population=None):
        self.state = state
        self.country = country
        self.county = county
        self.run_type = run_type
        self.population = population

        # define constants used in model parameter calculations
        self.observed_daily_growth_rate = 1.17
        self.days_to_model = 365

        # when going back to test hypothetical intervnetions in the past,
        # use this to start the data from this date instead of latest reported
        self.override_model_start = False

        ## Variables for calculating model parameters Hill -> our names/calcs
        # IncubPeriod: Average incubation period, days - presymptomatic_period
        # DurMildInf: Average duration of mild infections, days - duration_mild_infections
        # FracMild: Average fraction of (symptomatic) infections that are mild - (1 - hospitalization_rate)
        # FracSevere: Average fraction of (symptomatic) infections that are severe - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # FracCritical: Average fraction of (symptomatic) infections that are critical - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # CFR: Case fatality rate (fraction of infections that eventually result in death) - case_fatality_rate
        # DurHosp: Average duration of hospitalization (time to recovery) for individuals with severe infection, days - hospital_time_recovery
        # TimeICUDeath: Average duration of ICU admission (until death or recovery), days - icu_time_death

        # LOGIC ON INITIAL CONDITIONS:
        # hospitalized = case load from timeseries on last day of data / 4
        # mild = hospitalized / hospitalization_rate
        # icu = hospitalized * hospitalized_cases_requiring_icu_care
        # expoosed = exposed_infected_ratio * mild

        # Time before exposed are infectious (days)
        self.presymptomatic_period = 6

        # Time mildly infected people stay sick before
        # hospitalization or recovery (days)
        self.duration_mild_infections = 6

        # Time asymptomatically infected people stay
        # infected before recovery (days)
        self.duration_asymp_infections = 6

        # Duration of hospitalization before icu or
        # recovery (days)
        self.hospital_time_recovery = 11

        # Time from ICU admission to death (days)
        self.icu_time_death = 8

        ####################################################
        # BETA: transmission rate (new cases per day).
        # The rate at which infectious cases of various
        # classes cause secondary or new cases.
        ####################################################
        #
        # Transmission rate of infected people with no
        # symptoms [A] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.6
        # Current: Calculated based on observed doubling
        # rates
        #self.beta_asymp = 0.3 + ((self.observed_daily_growth_rate - 1.09) / 0.02) * 0.05
        self.beta_asymp = 0.4
        #
        # Transmission rate of infected people with mild
        # symptoms [I_1] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.6
        # Current: Calculated based on observed doubling
        # rates
        #self.beta_mild = 0.3 + ((self.observed_daily_growth_rate - 1.09) / 0.02) * 0.05
        self.beta_mild = 0.4
        #
        # Transmission rate of infected people with severe
        # symptoms [I_2] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.1
        self.beta_hospitalized = 0.1
        #
        # Transmission rate of infected people with severe
        # symptoms [I_3] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.1
        self.beta_icu = 0.1
        #
        ####################################################

        # Pecentage of asymptomatic, infectious [A] people
        # out of all those who are infected
        # make 0 to remove this stock
        self.percent_asymp = 0.3

        self.percent_infectious_symptomatic = 1 - self.percent_asymp

        self.hospitalization_rate = 0.10
        self.hospitalized_cases_requiring_icu_care = 0.25
        #self.hospitalized_cases_requiring_icu_care = 0.5

        self.percent_symptomatic_mild = (
            self.percent_infectious_symptomatic - self.hospitalization_rate
        )

        # changed this from CFR to make the calc of mu clearer
        self.death_rate_for_critical = 0.4

        # CFR is calculated from the input parameters vs. fixed
        self.case_fatality_rate = (
            (1 - self.percent_asymp)
            * self.hospitalization_rate
            * self.hospitalized_cases_requiring_icu_care
            * self.death_rate_for_critical
        )

        # if true we calculatied the exposed initial stock from the infected number vs. leaving it at 0
        self.exposed_from_infected = True
        self.exposed_infected_ratio = 1

        # different ways to model the actual data

        # cases represent all infected symptomatic
        # based on proportion of mild/hospitalized/icu
        # described in params
        # self.model_cases = "divided_into_infected"

        # 1/4 cases are hopsitalized, mild and icu
        # based on proporition of hopsitalized
        # described in params
        self.model_cases = "one_in_4_hospitalized"

        self.hospital_capacity_change_daily_rate = 1.05
        self.max_hospital_capacity_factor = 2.07
        self.initial_hospital_bed_utilization = 0.6
        self.case_fatality_rate_hospitals_overwhelmed = (
            self.hospitalization_rate * self.hospitalized_cases_requiring_icu_care
        )

        self.interventions = {}
        self.results_dict = OrderedDict()

    class SnapShot:
        def __init__(self, model_run, type):
            self.N = model_run.population

            # this is an inital run or a past run, will have to build the initial
            # conditions from the timeseries data
            if type in ("base", "past-counterfactual"):
                self.hospitalized = model_run.past_data.get(
                    key="estimated_hospitalized", default=0
                )
                self.icu = (
                    self.hospitalized * model_run.hospitalized_cases_requiring_icu_care
                )
                self.mild = (
                    model_run.past_data.get(key="active", default=0)
                    - self.hospitalized
                    - self.icu
                )
                self.asymp = self.mild * model_run.percent_asymp
                self.dead = model_run.past_data.get(key="deaths", default=0)

            elif type in ("intervention", "past-actual"):
                # this should be an intervention run, so the initial conditions are more
                # fleshed out
                self.mild = model_run.past_data.get(key="infected_a", default=0)
                self.hospitalized = model_run.past_data.get(key="infected_b", default=0)
                self.icu = model_run.past_data.get(key="infected_c", default=0)
                self.asymp = model_run.past_data.get(key="asymp", default=0)
                self.dead = model_run.past_data.get(key="dead", default=0)

            self.exposed = model_run.exposed_infected_ratio * self.mild
            self.infected = self.asymp + self.mild + self.hospitalized + self.icu
            self.recovered = model_run.past_data.get(key="recovered", default=0)
            susceptible = self.N - (self.infected + self.recovered + self.dead)

            self.y0 = [
                int(self.exposed),
                int(self.mild),
                int(self.hospitalized),
                int(self.icu),
                int(self.recovered),
                int(self.dead),
                int(self.asymp),
            ]

    def get_data(
        self, timeseries,
    ):
        # TODO rope in counties
        timeseries_df = pd.read_json(json.dumps(timeseries), orient='index')
        timeseries_df.index.name = 'date'
        timeseries_df.sort_index(inplace=True)
        timeseries_df.reset_index(inplace=True)

        if self.county is None:
            if self.population is None:
                if self.run_type == "city":
                    self.population = population[self.state]['urban_pop']
                else:
                    self.population = population[self.state]['rest_of_country']

            #self.timeseries = get_backfill_historical_estimates(timeseries_df)
            self.timeseries = timeseries_df
        else:
            # do county thing
            pass

        if self.override_model_start is False:
            self.start_date = self.timeseries.loc[
                (self.timeseries["cases"] > 0), "date"
            ].max()
        else:
            self.start_date = self.override_model_start

        self.actuals = self.timeseries.loc[(self.timeseries.date <= self.start_date), :]

        self.actuals.date = self.actuals.date.dt.date

        self.raw_actuals = self.actuals.copy()

        # do this when we run the base run
        self.processed_actuals = None

        # get a series of the relevant row in the df
        self.past_data = self.timeseries.loc[
            (self.timeseries.date == self.start_date), :
        ].iloc[0]

        self.default_past_data = self.past_data.copy()

        return

    def set_epi_model(self, epi_model_type):
        self.epi_run = EpiRun("base", self)
        self.epi_run.generate_epi_params()
        self.epi_run.InitConditions = self.SnapShot(self, "base")

    def reload_params(self):
        self.results_dict = OrderedDict()
        self.display_df = None

        self.actuals = self.raw_actuals

        self.epi_run.EpiParameters = None
        self.epi_run.InitConditions = None

        self.past_data = self.default_past_data

        self.epi_run.InitConditions = self.SnapShot(self, "base")

        self.epi_run.generate_epi_params()

    def run(self):
        self.epi_run.seir()
        self.epi_run.dataframe_ify()

        self.results_dict["base_run"] = self.epi_run.display_df.copy()

    def add_intervention(self, intervention: Intervention):
        intervention_name = f"intervention_{self.state}_{intervention.name}"

        intervention.system_name = intervention_name

        intervention_run = InterventionRun(intervention, self)
        intervention_run.InitConditions = self.SnapShot(
            self, intervention.intervention_type
        )

        self.interventions[intervention.system_name] = intervention_run

    def run_all_interventions(self):
        for intervention in self.interventions.values():
            if isinstance(intervention.intervention_start_date, datetime.datetime):
                intervention.intervention_start_date = (
                    intervention.intervention_start_date.date()
                )

        sorted_interventions = sorted(
            list(self.interventions.values()),
            key=lambda intervention: intervention.intervention_start_date,
        )

        for intervention in sorted_interventions:
            self.run_intervention(intervention.system_name)

    def get_prior_run(self, name):
        prior_run = list(self.results_dict.values())[-1].copy()

        return prior_run

    def run_intervention(self, name):
        # set the initial conditions based on the prior run
        self.interventions[name].set_prior_run(self.get_prior_run(name))

        self.past_data = self.interventions[name].initial_conditions

        _logger.error(self.interventions[name])

        self.interventions[name].load_epi()
        self.interventions[name].InitConditions = self.SnapShot(
            self, self.interventions[name].type
        )

        self.interventions[name].seir()
        self.interventions[name].dataframe_ify()

        self.results_dict[name] = self.interventions[name].display_df.copy()

        # if it's a counterfactual, move it to to the front of the line
        # to ensure it doesn't screw up later interventions
        if self.interventions[name].type == 'past-counterfactual':
            self.results_dict.move_to_end(name, last=False)


    def drop_interventions(self):
        self.interventions = {}
        self.run()

    def get_results(self):
        return self.results_dict

    def report(self, modelname):
        report_dict = {
            'modelrun': modelname,
            'state': self.state,
            'country': self.country,
            'county': self.county,
            'date': self.start_date,
            'interventions': [i.report() for i in self.interventions.values()],
        }

        return report_dict

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
