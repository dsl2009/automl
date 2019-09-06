import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv('data/weather.csv')
df.columns=['id',  'loc_id', 'mointer_name',
       'year', 'month', 'date','light_time',
       'wind1', 'wind2', 'wind3', 'wind4', 'wind_speed', 'rain', 'tem_high',
       'tem_low', 'tem_ave', 'humidity', 'pa']

df = df.replace('/',np.nan)
df = df.dropna(axis=0,how='any')
df = df.replace('*',np.nan)
df = df.dropna(axis=0,how='any')
sub_df = pd.DataFrame(df, columns=['loc_id', 'year', 'month', 'date','light_time',
                                   'rain', 'tem_high', 'tem_low', 'tem_ave', 'humidity'])

sub_df['tem_ave'].astype('float')




sub_df[ 'month'].astype('int')

sub_df = sub_df[(sub_df['month']>=3)&(sub_df['month']<=7)]
for name,data in sub_df.groupby(['loc_id','year']):
       for mon,dts in data.groupby('month'):
              print(mon,dts)

