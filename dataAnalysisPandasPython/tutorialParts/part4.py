import pandas as pd
import quandl

statesHMTL = pd.read_html('states.html')[0][1][1:]
quandlKeys = []


for abbreviation in statesHMTL:
    quandlKeys.append('resources/FMAC/HPI_'+abbreviation)



