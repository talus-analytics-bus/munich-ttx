"""Implementation of an SEIR compartment model with R = Recovered + Deceased,
I = I1 + I2 + I3 (increasing severity of infection), with an asymptomatic
infectious compartment (A).
"""

# standard modules
import datetime
import logging

# 3rd party modules
import numpy as np
import pandas as pd
from scipy.integrate import odeint

_logger = logging.getLogger(__name__)


def brute_force_r0(seir_params, new_r0, r0, N):
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
    calc_r0 = r0

    change = np.sign(new_r0 - calc_r0) * 0.00005
    # step = 0.1
    # direction = 1 if change > 0 else -1

    new_seir_params = seir_params.copy()

    while round(new_r0, 4) != round(calc_r0, 4):
        new_seir_params["beta"] = [
            0.0,
            new_seir_params["beta"][1] + change,
            new_seir_params["beta"][2],
            new_seir_params["beta"][3],
        ]
        calc_r0 = generate_r0(new_seir_params, N)

        diff_r0 = new_r0 - calc_r0

        # if the sign has changed, we overshot, turn around with a smaller
        # step
        if np.sign(diff_r0) != np.sign(change):
            change = -change / 2

    new_seir_params["beta"] = L(new_seir_params["beta"])

    return new_seir_params


def dataframe_ify(data, start, end, steps):
    """Return human-friendly dataframe of model results.

    Parameters
    ----------
    data : type
        Description of parameter `data`.
    start : type
        Description of parameter `start`.
    end : type
        Description of parameter `end`.
    steps : type
        Description of parameter `steps`.

    Returns
    -------
    type
        Description of returned object.

    """
    last_period = start + datetime.timedelta(days=(steps - 1))

    timesteps = pd.date_range(
        # start=start, end=last_period, periods=steps, freq=='D',
        start=start,
        end=last_period,
        freq="D",
    ).to_list()

    sir_df = pd.DataFrame(
        zip(data[0], data[1], data[2], data[3], data[4], data[5], data[6]),
        # zip(data[0], data[1], data[2], data[3], data[4], data[5]),
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
    sir_df.resample("1D").sum()

    # drop anything after the end day
    sir_df = sir_df.loc[:end]

    return sir_df


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
def deriv(y0, t, beta, alpha, gamma, rho, mu, f, N):
    """Calculate and return the current values of dE/dt, etc. for each model
    compartment as numerical integration is performed. This function is the
    first argument of the odeint numerical integrator function.

    Parameters
    ----------
    y0 : type
        Description of parameter `y0`.
    t : type
        Description of parameter `t`.
    beta : type
        Description of parameter `beta`.
    alpha : type
        Description of parameter `alpha`.
    gamma : type
        Description of parameter `gamma`.
    rho : type
        Description of parameter `rho`.
    mu : type
        Description of parameter `mu`.
    N : type
        Description of parameter `N`.

    Returns
    -------
    type
        Description of returned object.

    """
    # S = N - sum(y0)
    S = np.max([N - sum(y0), 0])

    E = y0[0]
    I1 = y0[1]
    I2 = y0[2]
    I3 = y0[3]
    R = y0[4]
    D = y0[5]
    A = y0[6]

    I_all = [I1, I2, I3]
    I_transmission = np.dot(beta[1:4], I_all)
    I_recovery = np.dot(gamma[1:4], I_all)
    A_transmission = A * beta.A
    A_recovery = A * gamma.A
    all_infected = sum(I_all) + A

    dE = np.min([(A_transmission + I_transmission) * S, S]) - (alpha * E)  # Exposed
    dA = ((1 - f) * alpha * E) - (gamma.A * A)  # asymp
    dI1 = (f * alpha * E) - (gamma[1] + rho[1]) * I1  # Ia - Mildly ill
    dI2 = (rho[1] * I1) - (gamma[2] + rho[2]) * I2  # Ib - Hospitalized
    dI3 = (rho[2] * I2) - ((gamma[3] + mu) * I3)  # Ic - ICU
    dR = np.min([A_recovery + I_recovery, all_infected])  # Recovered
    dD = mu * I3  # Deaths

    dy = [dE, dI1, dI2, dI3, dR, dD, dA]
    return dy


# Sets up and runs the integration
# start date and end date give the bounds of the simulation
# pop_dict contains the initial populations
# beta = contact rate
# gamma = mean recovery rate
# TODO: add other params from doc
def seir(pop_dict, model_parameters, beta, alpha, gamma, rho, mu, f):
    """Short summary.

    Parameters
    ----------
    pop_dict : type
        Description of parameter `pop_dict`.
    model_parameters : type
        Description of parameter `model_parameters`.
    beta : type
        Description of parameter `beta`.
    alpha : type
        Description of parameter `alpha`.
    gamma : type
        Description of parameter `gamma`.
    rho : type
        Description of parameter `rho`.
    mu : type
        Description of parameter `mu`.

    Returns
    -------
    type
        Description of returned object.

    """
    N = pop_dict["total"]

    # this should be an intervention run, so the initial conditions are more
    # fleshed out
    if "infected_b" in pop_dict:
        mild = pop_dict["infected_a"]
        hospitalized = pop_dict["infected_b"]
        icu = pop_dict["infected_c"]
        asymp = pop_dict["asymp"]
    # this is an inital run, will have to build the initial conditions from the
    # timeseries data
    else:
        hospitalized = pop_dict["infected"] / 4
        mild = hospitalized / model_parameters["hospitalization_rate"]
        icu = hospitalized * model_parameters["hospitalized_cases_requiring_icu_care"]
        asymp = mild * (
            model_parameters["percent_asymp"] / 1 - model_parameters["percent_asymp"]
        )

    exposed = model_parameters["exposed_infected_ratio"] * mild

    susceptible = pop_dict["total"] - (
        pop_dict["infected"] + pop_dict["recovered"] + pop_dict["deaths"]
    )

    y0 = [
        int(exposed),
        int(mild),
        int(hospitalized),
        int(icu),
        int(pop_dict.get("recovered", 0)),
        int(pop_dict.get("deaths", 0)),
        int(asymp),
    ]

    steps = 365
    t = np.arange(0, steps, 1)

    ret = odeint(deriv, y0, t, args=(beta, alpha, gamma, rho, mu, f, N))

    return np.transpose(ret), steps, ret


# # for testing purposes, just load the Harvard output
# def harvard_model_params(N):
#     return {
#         "beta": [0.0, 0.5 / N, 0.1 / N, 0.1 / N],
#         "alpha": 0.2,
#         "gamma": [0.0, 0.133, 0.125, 0.075],
#         "rho": [0.0, 0.033, 0.042],
#         "mu": 0.05,
#     }
#
#
# # for testing purposes, just load the Harvard output
# def r0_24_params(N):
#     return {
#         "beta": [0.0, 0.3719985820912413 / N, 0.1 / N, 0.1 / N],
#         "alpha": 0.2,
#         "gamma": [0.0, 0.133, 0.125, 0.075],
#         "rho": [0.0, 0.033, 0.042],
#         "mu": 0.05,
#     }


# for now just implement Harvard model, in the future use this to change
# key params due to interventions
def generate_epi_params(model_parameters):
    N = model_parameters["population"]

    fraction_critical = (
        (1 - model_parameters["percent_asymp"])
        * model_parameters["hospitalization_rate"]
        * model_parameters["hospitalized_cases_requiring_icu_care"]
    )

    alpha = 1 / model_parameters["presymptomatic_period"]

    beta = L(
        0,
        model_parameters["beta"] / N,
        model_parameters["beta_hospitalized"] / N,
        model_parameters["beta_icu"] / N,
        # TODO move beta.A to model params
        A=model_parameters["beta"] / N,
        # A = 0,
    )

    # have to calculate these in order and then put them into arrays
    gamma_0 = 0
    gamma_1 = (1 / model_parameters["duration_mild_infections"]) * (
        1 - model_parameters["hospitalization_rate"]
    )

    rho_0 = 0
    rho_1 = (1 / model_parameters["duration_mild_infections"]) - gamma_1
    rho_2 = (1 / model_parameters["hospital_time_recovery"]) * (
        fraction_critical / model_parameters["hospitalization_rate"]
    )

    gamma_2 = (1 / model_parameters["hospital_time_recovery"]) - rho_2

    mu = (1 / model_parameters["icu_time_death"]) * (
        model_parameters["case_fatality_rate"] / fraction_critical
    )
    gamma_3 = (1 / model_parameters["icu_time_death"]) - mu

    seir_params = {
        "beta": beta,
        "alpha": alpha,
        # TODO move gamma_a to model params
        "gamma": L(gamma_0, gamma_1, gamma_2, gamma_3, A=gamma_1),
        # "gamma": L(gamma_0, gamma_1, gamma_2, gamma_3, A = 0),
        "rho": [rho_0, rho_1, rho_2],
        "mu": mu,
        "f": model_parameters["percent_asymp"],
    }

    return seir_params


# for now just implement Harvard model, in the future use this to change
# key params due to interventions
def generate_epi_params_from_mr(model_run):
    N = model_run.population

    fraction_critical = (
        (1 - model_run.percent_asymp)
        * model_run.hospitalization_rate
        * model_run.hospitalized_cases_requiring_icu_care
    )

    alpha = 1 / model_run.presymptomatic_period

    beta = L(
        0,
        model_run.beta / N,
        model_run.beta_hospitalized / N,
        model_run.beta_icu / N,
        # TODO move beta.A to model params
        A=model_run.beta / N,
        # A = 0,
    )

    # have to calculate these in order and then put them into arrays
    gamma_0 = 0
    gamma_1 = (1 / model_run.duration_mild_infections) * (
        1 - model_run.hospitalization_rate
    )

    rho_0 = 0
    rho_1 = (1 / model_run.duration_mild_infections) - gamma_1
    rho_2 = (1 / model_run.hospital_time_recovery) * (
        fraction_critical / model_run.hospitalization_rate
    )

    gamma_2 = (1 / model_run.hospital_time_recovery) - rho_2

    mu = (1 / model_run.icu_time_death) * (
        model_run.case_fatality_rate / fraction_critical
    )
    gamma_3 = (1 / model_run.icu_time_death) - mu

    seir_params = {
        "beta": beta,
        "alpha": alpha,
        # TODO move gamma_a to model params
        "gamma": L(gamma_0, gamma_1, gamma_2, gamma_3, A=gamma_1),
        # "gamma": L(gamma_0, gamma_1, gamma_2, gamma_3, A = 0),
        "rho": [rho_0, rho_1, rho_2],
        "mu": mu,
        "f": model_run.percent_asymp,
    }

    return seir_params


def convert_ratio_to_frac(x, numerator="second_term"):
    """Given a ratio x where x = a / b, returns the corresponding fraction
    assuming that b is the numerator by default.
    Parameters
    ----------
    x : float
        ratio where x = a /b.
    numerator : string
        if 'second_term': returns fraction assuming b is numerator.
        if 'first_term': returns fraction assuming a is numerator.
    Returns
    -------
    float
        fraction assuming b is the numerator by default, i.e., b / (a + b).
        if 'first_term' was specified as `numerator` then returns fraction
            assuming a is the numerator, i.e., a / (a + b)
    """
    if numerator == "second_term":
        return 1 / (x + 1)
    else:
        return x / (x + 1)


# TODO update to match latest model:
# R0 = N*((1-f)*BA/gA + f*((B1/(p1+g1))+(p1/(p1+g1))*(B2/(p2+g2)+ (p2/(p2+g2))*(B3/(m+g3)))))
def generate_r0(seir_params, N):
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
    b = seir_params["beta"]
    p = seir_params["rho"]
    g = seir_params["gamma"]
    u = seir_params["mu"]

    r0 = N * (
        (b[1] / (p[1] + g[1]))
        + (p[1] / (p[1] + g[1]))
        * (b[2] / (p[2] + g[2]) + (p[2] / (p[2] + g[2])) * (b[3] / (u + g[3])))
    )

    return r0


# Credit: http://code.activestate.com/recipes/579103-python-addset-attributes-to-list/
class L(list):
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.

    The problem:
    a = [1, 2, 4, 8]
    a.x = "Hey!" # AttributeError: 'list' object has no attribute 'x'

    The solution:
    a = L(1, 2, 4, 8)
    a.x = "Hey!"
    print a       # [1, 2, 4, 8]
    print a.x     # "Hey!"
    print len(a)  # 4

    You can also do these:
    a = L( 1, 2, 4, 8 , x="Hey!" )                 # [1, 2, 4, 8]
    a = L( 1, 2, 4, 8 )( x="Hey!" )                # [1, 2, 4, 8]
    a = L( [1, 2, 4, 8] , x="Hey!" )               # [1, 2, 4, 8]
    a = L( {1, 2, 4, 8} , x="Hey!" )               # [1, 2, 4, 8]
    a = L( [2 ** b for b in range(4)] , x="Hey!" ) # [1, 2, 4, 8]
    a = L( (2 ** b for b in range(4)) , x="Hey!" ) # [1, 2, 4, 8]
    a = L( 2 ** b for b in range(4) )( x="Hey!" )  # [1, 2, 4, 8]
    a = L( 2 )                                     # [2]
    """

    def __new__(self, *args, **kwargs):
        return super(L, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
