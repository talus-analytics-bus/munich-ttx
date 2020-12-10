"""Implementation of an SEIR compartment model with R = Recovered + Deceased,
I = I1 + I2 + I3 (increasing severity of infection), with an asymptomatic
infectious compartment (A).
"""

# standard modules
import datetime
import logging
import math
from copy import deepcopy

# 3rd party modules
import numpy as np
import pandas as pd
from scipy.integrate import odeint

_logger = logging.getLogger(__name__)


# class TalusSEIR(EpiRun):
class TalusSEIR:
    def __init__(self, type, model_run):
        self.initlization_time = datetime.datetime.now()
        self.type = type  # base, past, or intervention
        self.start_date = model_run.start_date
        self.model_run = model_run

    def generate_epi_params(self):
        self.EpiParameters = self.EpiParams(self.model_run)

    class EpiParams:
        def __init__(self, model_run):

            model_run.case_fatality_rate = (
                (1 - model_run.percent_asymp)
                * model_run.hospitalization_rate
                * model_run.hospitalized_cases_requiring_icu_care
                * model_run.death_rate_for_critical
            )

            self.N = model_run.population

            self.fraction_critical = (
                (1 - model_run.percent_asymp)
                * model_run.hospitalization_rate
                * model_run.hospitalized_cases_requiring_icu_care
            )

            self.alpha = 1 / model_run.presymptomatic_period

            self.beta = (
                0,
                model_run.beta_mild / self.N,
                model_run.beta_hospitalized / self.N,
                model_run.beta_icu / self.N,
                # TODO move beta.A to model params
                model_run.beta_mild / self.N,
            )

            self.beta_mild = model_run.beta_mild
            self.beta_asymp = model_run.beta_asymp

            self.beta_A = model_run.beta_mild / self.N

            # have to calculate these in order and then put them into arrays
            self.gamma_0 = 0
            self.gamma_1 = (1 / model_run.duration_mild_infections) * (
                1 - model_run.hospitalization_rate
            )

            self.rho_0 = 0
            self.rho_1 = (1 / model_run.duration_mild_infections) - self.gamma_1
            self.rho_2 = (1 / model_run.hospital_time_recovery) * (
                self.fraction_critical / model_run.hospitalization_rate
            )

            self.gamma_2 = (1 / model_run.hospital_time_recovery) - self.rho_2

            self.mu = (1 / model_run.icu_time_death) * (
                model_run.case_fatality_rate / self.fraction_critical
            )
            self.gamma_3 = (1 / model_run.icu_time_death) - self.mu

            # TODO move gamma_a to model params
            self.gamma_A = self.gamma_1

            self.gamma = (
                self.gamma_0,
                self.gamma_1,
                self.gamma_2,
                self.gamma_3,
                self.gamma_A,
            )
            # "gamma": L(gamma_0, gamma_1, gamma_2, gamma_3, A = 0),
            self.rho = [self.rho_0, self.rho_1, self.rho_2]
            self.f = model_run.percent_asymp

            # these are used to get the growth-rate-base R estimation
            # Serial interval =[Incubation Period]+1/2[Duration of mild infection]
            self.serial_interval = model_run.presymptomatic_period + (
                model_run.duration_mild_infections / 2
            )
            # f = Ratio of mean infectious period to mean serial interval
            self.f_ratio = (
                model_run.duration_mild_infections / 2
            ) / self.serial_interval

        def report(self):
            return {
                'beta_mild': self.beta_mild,
                'beta_asymp': self.beta_asymp,
            }

        # TODO update to match latest model:
        # R0 = N*((1-f)*BA/gA + f*((B1/(p1+g1))+(p1/(p1+g1))*(B2/(p2+g2)+ (p2/(p2+g2))*(B3/(m+g3)))))
        def generate_r0(self):
            """Short summary.

            Parameters
            ----------
            seir_params : type
                Description of parameter `seir_params`.
            N : type
                Description of parameter `N`.

            Returns
            -------
            type
                Description of returned object.

            """
            b = self.beta
            p = self.rho
            g = self.gamma
            u = self.mu

            r0 = self.N * (
                (b[1] / (p[1] + g[1]))
                + (p[1] / (p[1] + g[1]))
                * (b[2] / (p[2] + g[2]) + (p[2] / (p[2] + g[2])) * (b[3] / (u + g[3])))
            )

            return r0

    def process_actuals(self, actuals):
        actuals.loc[:, "infected_b"] = actuals.loc[:, "estimated_hospitalized"]

        actuals.loc[:, "infected_c"] = (
            actuals.loc[:, "infected_b"] * self.model_run.hospitalized_cases_requiring_icu_care
        )
        actuals.loc[:, "infected_a"] = (
            actuals.loc[:, "active"]
            - actuals.loc[:, "infected_b"]
            - actuals.loc[:, "infected_c"]
        )

        actuals.loc[:, "exposed"] = (
            self.model_run.exposed_infected_ratio * actuals["infected_a"]
        )

        actuals.loc[:, "recovered"] = actuals["estimated_recovered"]

        actuals.loc[:, "dead"] = actuals["deaths"]

        actuals.loc[:, "asymp"] = (
            actuals["infected_a"] * self.model_run.percent_asymp
        )

        match_columns = [
            "date",
            "exposed",
            "infected_a",
            "infected_b",
            "infected_c",
            "recovered",
            "dead",
            "asymp",
        ]

        return actuals.loc[:, match_columns].copy()

    def dataframe_ify(self):
        """Generate human-friendly dataframe of model results and combine with past
        results

        Parameters
        ----------
        self : EpiRun object

        Returns
        -------
        type
            Description of returned object.

        """
        if self.type == "base":
            run_start_date = self.start_date
        else:
            run_start_date = self.model_start_date

        self.last_period = run_start_date + datetime.timedelta(days=(self.steps - 1))

        timesteps = pd.date_range(
            # start=start, end=last_period, periods=steps, freq=='D',
            start=run_start_date,
            end=self.last_period,
            freq="D",
        ).to_list()

        data = self.results

        seir_df = pd.DataFrame(
            zip(data[0], data[1], data[2], data[3], data[4], data[5], data[6]),
            columns=[
                "exposed",
                "infected_a",
                "infected_b",
                "infected_c",
                "recovered",
                "dead",
                "asymp",
            ],
            index=timesteps,
        )

        # reample the values to be daily
        seir_df.resample("1D").sum()

        # drop anything after the end day
        seir_df = seir_df.loc[: self.last_period]
        seir_df.index = seir_df.index.copy().date
        seir_df.index.name = "date"
        seir_df.reset_index(inplace=True)

        r_effective = self.EpiParameters.generate_r0()

        seir_df["R effective"] = r_effective

        # if there is a past run, get that info
        if self.type == "base":
            actual_df = self.process_actuals(self.model_run.raw_actuals)
            actual_df["source"] = "actuals"

            # use the base R for the actuals as well (so the plots line up)
            actual_df["R effective"] = self.EpiParameters.generate_r0()

            self.model_run.processed_actuals = actual_df

            seir_df["source"] = "base run"
            display_df = actual_df.append(seir_df)

        else:

            past_run = self.prior_results.loc[
                (self.prior_results["date"] < run_start_date)
            ]

            if self.type == "past-actual":
                intervention_start_date = self.intervention_start_date
                past_run.loc[
                    (past_run["date"] >= intervention_start_date), "R effective"
                ] = r_effective

            seir_df["source"] = self.name

            _logger.error(seir_df["source"])

            display_df = past_run.append(seir_df)

        self.results_df = seir_df

        display_df["infected"] = (
            display_df.infected_a
            + display_df.infected_b
            + display_df.infected_c
            + display_df.asymp
        )

        # pct_change is really lumpy, get rolling 7 days
        display_df["pct_change"] = (
            display_df.loc[:, "infected_a"].pct_change().rolling(7).mean()
        )
        display_df["doubling_time"] = math.log(2) / display_df["pct_change"]

        # estimated R from growth rate
        # Serial interval =[Incubation Period]+1/2[Duration of mild infection]
        # f = Ration of mean infectious period to mean serial interval
        # def estimateR(growth_rate):
        #    return (
        #        1
        #        + (self.EpiParameters.serial_interval * growth_rate)
        #        + (
        #            self.EpiParameters.f_ratio
        #            * (1 - self.EpiParameters.f_ratio)
        #            * ((self.EpiParameters.serial_interval * growth_rate) ** 2)
        #        )
        #    )

        # display_df["estimated_R"] = display_df["pct_change"].apply(estimateR)
        #six_months = (datetime.datetime.now() + datetime.timedelta(days=180)).date()
        #display_df = display_df.loc[display_df.date <= six_months, :].copy()

        #six_months = (datetime.datetime.now() + datetime.timedelta(days=180)).date()
        #display_df = display_df.loc[display_df.date <= six_months, :].copy()

        self.display_df = display_df

        return

    # The SEIR model differential equations.
    # https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb
    # but these are the basics
    # y = initial conditions
    # t = a grid of time points (in days) - not currently used, but will be for time-dependent functions
    # N = total pop
    # beta = contact rate
    # gamma = mean recovery rate
    # Don't track S because all variables must add up to 1
    # include blank first entry in vector for beta, gamma, p so that indices align in equations and code.
    # In the future could include recovery or infection from the exposed class (asymptomatics)
    def deriv(self, y0, t):
        """Calculate and return the current values of dE/dt, etc. for each model
        compartment as numerical integration is performed. This function is the
        first argument of the odeint numerical integrator function.

        Parameters
        ----------
        y0 : type
            Description of parameter `y0`.
        t : type
            Description of parameter `t`.

        Returns
        -------
        type


        """
        # S = N - sum(y0)
        S = np.max([self.model_run.population - sum(y0), 0])

        E = y0[0]
        I1 = y0[1]
        I2 = y0[2]
        I3 = y0[3]
        R = y0[4]
        D = y0[5]
        A = y0[6]

        I_all = [I1, I2, I3]
        I_transmission = np.dot(self.EpiParameters.beta[1:4], I_all)
        I_recovery = np.dot(self.EpiParameters.gamma[1:4], I_all)
        A_transmission = A * self.EpiParameters.beta_A
        A_recovery = A * self.EpiParameters.gamma_A
        all_infected = sum(I_all) + A
        percent_asymp = self.EpiParameters.f

        dE = np.min([(A_transmission + I_transmission) * S, S]) - (
            self.EpiParameters.alpha * E
        )  # Exposed
        dA = (percent_asymp * self.EpiParameters.alpha * E) - (
            self.EpiParameters.gamma_A * A
        )  # asymp
        dI1 = ((1 - percent_asymp) * self.EpiParameters.alpha * E) - (
            self.EpiParameters.gamma[1] + self.EpiParameters.rho[1]
        ) * I1  # Ia - Mildly ill
        dI2 = (self.EpiParameters.rho[1] * I1) - (
            self.EpiParameters.gamma[2] + self.EpiParameters.rho[2]
        ) * I2  # Ib - Hospitalized
        dI3 = (self.EpiParameters.rho[2] * I2) - (
            (self.EpiParameters.gamma[3] + self.EpiParameters.mu) * I3
        )  # Ic - ICU
        dR = np.min([A_recovery + I_recovery, all_infected])  # Recovered
        dD = self.EpiParameters.mu * I3  # Deaths

        dy = [dE, dI1, dI2, dI3, dR, dD, dA]
        return dy

    # Sets up and runs the integration
    # start date and end date give the bounds of the simulation
    # pop_dict contains the initial populations
    # beta = contact rate
    # gamma = mean recovery rate
    # TODO: add other params from doc
    def seir(self):
        self.steps = self.model_run.days_to_model
        t = np.arange(0, self.steps, 1)

        y0 = [
            self.InitConditions.exposed,
            self.InitConditions.mild,
            self.InitConditions.hospitalized,
            self.InitConditions.icu,
            self.InitConditions.recovered,
            self.InitConditions.dead,
            self.InitConditions.asymp,
        ]

        ret = odeint(self.deriv, y0, t, args=())

        self.results = np.transpose(ret)

        return

    def epi_report(self):
        # TODO: blow this out... summary of key assumptions
        return

    def model_report(self):
        # TODO: blow this out... summary of key assumptions
        return

    def param_report(self):
        model_report = self.model_report()
        epi_report = self.epi_report()

        return (model_report, epi_report)


class InterventionRun(TalusSEIR):
    def __init__(self, intervention, model_run):
        # get the base run done so we have a place to start
        self.type = intervention.intervention_type
        self.model_run = model_run
        self.intervention = intervention
        self.startdate = intervention.startdate
        self.description = intervention.description
        self.model_start_date = intervention.model_start_date
        self.intervention_start_date = intervention.intervention_start_date
        self.name = intervention.name
        self.system_name = intervention.system_name

    def load_epi(self):
        super(InterventionRun, self).__init__(self.type, self.model_run)
        self.get_new_epi_params()

    def get_new_epi_params(self):
        for param_name, param_value in self.intervention.params:

            if param_name == "r0":
                self.model_run.beta_mild = self.brute_force_r0(param_value)
            else:
                setattr(self.model_run, param_name, param_value)

        self.EpiParameters = self.EpiParams(self.model_run)

    def set_prior_run(self, prior_run):
        # if it's a future intervention, use the projections from that date
        # if it's a past intervention that actually took place, we start from the
        # same initial conditions as the base run

        #_logger.error('checking prior_run')
        #_logger.error(self.model_start_date)
        #_logger.error(prior_run)

        if self.type in ("intervention", "past-actual"):
            self.initial_conditions = prior_run.loc[
                (prior_run["date"] == self.model_start_date)
            ].iloc[0]

            self.prior_results = prior_run

        # if it's a past intervention that did not take place, start from the
        # best data we have on the day of that intervention
        elif self.type == "past-counterfactual":
            self.initial_conditions = self.model_run.actuals[
                (self.model_run.actuals.date == self.intervention_start_date)
            ].iloc[0]

            self.prior_results = self.model_run.processed_actuals

    def report(self):
        return {
            'name': self.name,
            'system_name': self.system_name,
            'intervention_type': self.type,
            'description': self.description,
            'startdate': self.startdate,
            'model_start_date': self.model_start_date,
            'intervention_start_date': self.intervention_start_date,
            'params': self.EpiParameters.report(),
        }

    def brute_force_r0(self, new_r0):
        """This function will be obsolete when the procedure for introducing
        interventions into model runs is updated -- do not maintain it.

        Parameters
        ----------
        seir_params : type
            Description of parameter `seir_params`.
        new_r0 : type
            Description of parameter `new_r0`.
        r0 : type
            Description of parameter `r0`.
        N : type
            Description of parameter `N`.

        Returns
        -------
        type
            Description of returned object.

        """
        calc_r0 = self.r0

        change = np.sign(new_r0 - calc_r0) * 0.00005
        # step = 0.1
        # direction = 1 if change > 0 else -1

        NewEpiParams = self.EpiParameters.deepcopy()

        while round(new_r0, 4) != round(calc_r0, 4):
            NewEpiParams["beta"] = [
                0.0,
                NewEpiParams["beta"][1] + change,
                NewEpiParams["beta"][2],
                NewEpiParams["beta"][3],
            ]
            calc_r0 = generate_r0(new_seir_params, N)

            diff_r0 = new_r0 - calc_r0

            # if the sign has changed, we overshot, turn around with a smaller
            # step
            if np.sign(diff_r0) != np.sign(change):
                change = -change / 2

        new_seir_params["beta"] = L(new_seir_params["beta"])

        return new_seir_params
