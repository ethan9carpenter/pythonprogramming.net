import pandas as pd
import quandl

#quandl.get("ZILLOW/Z77006_ZRISFRR")
data  = pd.read_csv('ZILLOW-Z77006_ZRISFRR.csv', index_col=0)
#data = pd.read_csv('ZILLOW-Z77006_ZRISFRR.csv', index_col=0, names=['House Price', 'Date'])

#data.set_index("Date", inplace=True)
data.columns = ['Median_Rent']

data.rename(columns={'Median_Rent': 'Rent'}, inplace=True)
#data.to_csv("newcsv2.csv")
#data.to_html("firstHTML.html")

print(data.head())