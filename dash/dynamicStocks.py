import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
from datetime import datetime as dt
import json
import quandl

with open('/Users/footballnerd12/apiKeys.json', 'r') as file:
    dict = json.load(file)
    quandl.ApiConfig.api_key =  dict['quandl']

app = dash.Dash()
app.config['suppress_callback_exceptions']=True

def buildGraph():
    df = pd.read_pickle('practiceRaceTable.pickle')
    df.drop(['Place', 'Bib', 'Lane', 'Affiliation', 'Time'], axis=1, inplace=True)
    
    splits = df.drop('Athlete', axis=1)
    runners = df['Athlete']
    
    lines = []
    
    for i, row in splits.iterrows():
        data = {'x': list(range(1, 1+len(row))), 'y': row, 'name': runners[i]}
        lines.append(data)
    graph = dcc.Graph(id='example', figure={'data': lines})

    return graph

def getStockGraph(ticker='AAPL', start=dt(2015, 1, 1)):
    end = dt.now()

    df = quandl.get('WIKI/'+str(ticker), start_date=start, end_date=end)
    x = df.index
    y = df['Close']

    line = {'x': x, 'y': y, 'name': ticker, 'type': 'line'}
    layout = {'title': ticker}
    
    return dcc.Graph(id='stock', figure={'data': [line], 'layout': layout})
    
#Children
inputTitle = html.Div('Ticker to Graph')
input = dcc.Input(id='input', value='', type='Text')
output = html.Div(id='output-graph')
children = inputTitle, input, output  

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='input', component_property='value')])
def updateGraph(inputData):
    return getStockGraph(inputData)

#App
app.layout = html.Div(children=children)                         

if __name__ == '__main__':
    app.run_server(debug=True)