import time
import numpy as np
import pandas as pd

from MODA_SVC import moda_svc
from NSGAII_SVC import nsga2_svc
from MOPSO_SVC import mopso_svc
from MOGWO_SVC import mogwo_svc
from IMOGWO_SVC import imogwo_svc
from IEMOEA_SVC import iemoea_svc
from ISNA_SVC import isna_svc

datasets_name = ['Parkinson']
arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', 'q7', '18', '19', '20',
       '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

for name in datasets_name:
    moda_svc(name, arr)
    nsga2_svc(name, arr)
    mopso_svc(name, arr)
    mogwo_svc(name, arr)
    imogwo_svc(name, arr)
    iemoea_svc(name, arr)
    isna_svc(name, arr)