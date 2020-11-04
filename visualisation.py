import pandas as pd
import plotly.express as px
import model

yhat = model.yhat
preds = model.preds

yhat.drop([yhat.columns[0], 'ID'], axis=1, inplace=True)

preds['Latitude'] = preds[preds.columns[1]]
preds['Longitude'] = preds[preds.columns[2]]
preds.drop([preds.columns[0], preds.columns[1], preds.columns[2]], axis=1, inplace=True)
preds['Prediction'] = 1

preoutputDf = yhat.append(preds, ignore_index=True)

origin = pd.DataFrame(
    data={'Latitude': preoutputDf.iloc[0]['Latitude'], 'Longitude': preoutputDf.iloc[0]['Longitude'], 'Prediction': 1},
    columns=['Latitude', 'Longitude', 'Prediction'], index=[len(yhat.index)])

outputDf = (yhat.append(origin, ignore_index=True)).append(preds, ignore_index=True)
outputDf['Prediction'] = outputDf['Prediction'].fillna(0)

fig = px.line_mapbox(outputDf, lat='Latitude', lon='Longitude', color='Prediction',
                     center=dict(lat=0, lon=180), zoom=0,
                     mapbox_style="stamen-terrain")
fig.show()
