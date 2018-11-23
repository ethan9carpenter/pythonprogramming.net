import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

df = pd.read_pickle('practiceRaceTable.pickle')
df.drop(['Place', 'Bib', 'Lane', 'Affiliation', 'Time'], axis=1, inplace=True)

splits = df.drop('Athlete', axis=1)
runners = df['Athlete']

lines = []

for i, row in splits.iterrows():
    data = {'x': list(1, 1+range(len(row))), 'y': row, 'name': runners[i]}
    lines.append(data)
   
graph = dcc.Graph(id='example', figure={'data': lines})

app = dash.Dash()
app.layout = html.Div(children=[
        html.H1('Race'),
        graph
    ])                         

"""
app = dash.Dash()
app.layout = html.Div(children=[
    html.H1('Dash Tutorial'),
    dcc.Graph(id='example',
              figure={
                  'data': [
                      {'x': [1, 2, 3, 4, 5], 'y': [0, -1, 5, 8, 10], 'type': 'line', 'name': 'votes'},
                      {'x': [1, 4, 7, 0, 2], 'y': [0, -1, 5, 8, 10], 'type': 'bar', 'name': 'names'}
                      ]
                  })
    ])
"""

if __name__ == '__main__':
    app.run_server(debug=True)