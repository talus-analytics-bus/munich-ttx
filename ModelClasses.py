from enum import Enum
from datetime import datetime, date
from typing import Set, List


from pydantic import BaseModel, Field

##
# Models for body parameters
##
class ModelTypes(str, Enum):
    base = "base"
    intervention = "intervention"
    past_actual = "past-actual"
    past_counterfactual = "past-counterfactual"


# start with beta,
class EpiParams(BaseModel):
    beta_mild: float = 0.5
    beta_asymp: float = 0.5


class Intervention(BaseModel):
    name: str
    system_name: str = None
    intervention_type = ModelTypes.base
    description: str = None
    startdate: date
    model_start_date: date = None
    intervention_start_date: date = None
    params: EpiParams

class Interventions(BaseModel):
    list: List[Intervention] = []

class DateReport(BaseModel):
    last_data_update: date = None
    last_policy_update: date = None



class Result(BaseModel):
    name: str
    run: str


class ModelReport(BaseModel):
    modelrun: str
    state: str
    population: int = 0
    date: date
    death_date: date
    cases: int = 0
    deaths: int = 0
    counterfactual_cases: int = 0
    counterfactual_deaths: int = 0
    results: List[Result] = []
    interventions: List[Intervention] = []


class StateQuery(BaseModel):
    state: str

class ModelQuery(BaseModel):
    model_id: str
