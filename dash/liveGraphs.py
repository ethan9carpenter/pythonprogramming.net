import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as grObjs
from collections import deque

X = deque(maxlen=100)
Y = deque(maxlen=100)
X.append(1)
Y.append(1)

app = dash.Dash(__name__)

#Layout
graph = dcc.Graph(id='graph', animate=True)
interval = dcc.Interval(id='update-interval', interval=100)
children = graph, interval

app.layout = html.Div(children=children)

@app.callback(output=Output('graph', 'figure'), 
              events=[Event('update-interval', 'interval')])
def updateGraph():
    global X
    global Y
    X.append(X[-1]+1)
    Y.append(Y[-1]*(1+random.uniform(-0.1, 0.1)))

    data = grObjs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode='lines+markers'
        )
    layout = grObjs.Layout(
            xaxis=dict(range=[min(X), max(X)]),
            yaxis=dict(range=[min(Y), max(Y)])
        )
    return {'data': [data], 
            'layout': layout}
    
    
if __name__ == '__main__':
    app.run_server(debug=True)   
    
    
    
    