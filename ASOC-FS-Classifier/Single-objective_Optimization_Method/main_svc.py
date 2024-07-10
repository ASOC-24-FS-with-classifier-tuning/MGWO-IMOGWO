import numpy as np
import pandas as pd
from DA_SVC import da_svc
from GA_SVC import ga_svc
from PSO_SVC import pso_svc
from GWO_SVC import gwo_svc
from MGWO_SVC import mgwo_svc
from ISPSO_SVC import ispso_svc
from SPACO_SVC import spaco_svc

datasets_name = ['Parkinson']
runTime = 30


for name in datasets_name:
    da_svc(name ,runTime)
    ga_svc(name, runTime)
    pso_svc(name, runTime)
    gwo_svc(name, runTime)
    mgwo_svc(name, runTime)
    ispso_svc(name, runTime)
    spaco_svc(name, runTime)