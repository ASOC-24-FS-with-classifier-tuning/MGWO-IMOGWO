import math
import numpy


def tentmap12(Initial_value, Parameters, Max_iter):
    x = Initial_value
    pops = []
    for i in range(Max_iter):
        if x < Parameters:
            x = x / Parameters
        else:
            x = (1 - x)/(1-Parameters)
        pops.append(x)
    return pops

def sinusoidalmap11(Initial_value, Parameters, Max_iter):
    x = Initial_value
    pops = []
    for i in range(Max_iter):
        x = Parameters * x ** 2 * math.sin(math.pi * x)
        pops.append(x)
    return pops

