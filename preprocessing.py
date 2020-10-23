import datetime
import random

import pandas as pd

# Had done some extra preprocessing, but saved to .csv file and removed from here

df = pd.read_csv("Data/hurricane_data.csv", index_col=0)

# Replacing categorical features with dummy variables

df.drop(df.columns[[11, 22]], axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
dummies = pd.get_dummies(df['Status'])
df = df.join(dummies.drop([0])).drop('Status', axis=1)
df2 = df[dummies.columns]
df2 = df2.drop(df2.index[0])
df[dummies.columns] = dummies

dummies = pd.get_dummies(df['Event'])
df = df.join(dummies.drop([0])).drop('Event', axis=1)
df2 = df[dummies.columns]
df2 = df2.drop(df2.index[0])
df[dummies.columns] = dummies

# Extracting features from date column for use with regression

df['Date'] = df['Date'].apply(lambda x: datetime.datetime.fromisoformat(x))
df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month)
df.drop('Date', inplace=True, axis=1)

y = df[['ID', 'Latitude', 'Longitude']].groupby('ID')
x = df.groupby('ID')

x_train, x_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for group in x.groups:
    if random.random() <= 0.4:
        x_test = x_test.append(x.get_group(group))
        y_test = y_test.append(y.get_group(group))
    else:
        x_train = x_train.append(x.get_group(group))
        y_train = y_train.append(y.get_group(group))

idx = x_train.groupby('ID').cumcount()
x_train = x_train[idx != 0].groupby('ID')
idx = x_test.groupby('ID').cumcount()
x_test = x_test[idx != 0].groupby('ID')

idx = y_train.groupby('ID').cumcount(ascending=False)
y_train = y_train[idx != 0].groupby('ID')
idx = y_test.groupby('ID').cumcount(ascending=False)
y_test = y_test[idx != 0].groupby('ID')
