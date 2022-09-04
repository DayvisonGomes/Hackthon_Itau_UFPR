import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def creation_train(cidades, dados, treino):
    ano_ = 0
    ano_2 = 0
    
    if treino == True:
        ano_ = 2004
        ano_2 = 2018
    
    else:
        ano_ = 2018
        ano_2 = 2021
        
    dic_ = {}
    
    for i in range(38):
        dic_[i] = [0]
    
    n_df = pd.DataFrame(data=dic_)
    n_df.columns = dados.columns
    
    df = dados[dados['codigo_ibge'] == dados['codigo_ibge'][0]]
        
    for ano in range(ano_,ano_2):
        df_ = df[df['data'] >= pd.Timestamp(ano,1,1,0)]
        df_2 = df_[df_['data'] < pd.Timestamp(ano+1,1,1,0)]
        n_df.loc[len(n_df)] = [ano, dados['codigo_ibge'][0]] + df_2.mean().tolist()[1:]
            
            
    for i in range(1,len(dados['codigo_ibge'])-1):
        
        if dados['codigo_ibge'][i] == dados['codigo_ibge'][i+1]:
            pass
        
        else:
            df = dados[dados['codigo_ibge'] == dados['codigo_ibge'][i+1]]
            
            for ano in range(ano_,ano_2):
                df_ = df[df['data'] >= pd.Timestamp(ano,1,1,0)]
                df_2 = df_[df_['data'] < pd.Timestamp(ano+1,1,1,0)]
                n_df.loc[len(n_df)] = [ano, dados['codigo_ibge'][i+1]] + df_2.mean().tolist()[1:]
    
    return n_df.drop(n_df.index[0])


def valores(dados, cidades):
    values = []
    
    for codigo in dados['codigo_ibge']:
        values = values + dados[dados['codigo_ibge'] == codigo].iloc[0, 3:17].tolist()
        
    return values


def series_to_supervised(serie, steps_in):
    features = pd.DataFrame()
    features['t'] = serie 
    
    for i in range(1, steps_in+1): 
        features['t-'+str(i)] = serie.shift(i)
    features = features.iloc[steps_in:] 
    
    return features


def cada_cidade_uma_linha(df):
    df_ = series_to_supervised(df[df['codigo_ibge']==4100103].set_index('data')['Production'],13)
    df_['codigo_ibge'] = 4100103
    
    for cidade in df['codigo_ibge'].unique()[1:]:
        df_2 = series_to_supervised(df[df['codigo_ibge']==cidade].set_index('data')['Production'],13)
        df_2['codigo_ibge'] = cidade
        df_ = pd.concat([df_,df_2])
        
    return df_


def comprensao(df):
    dic_ = {}
    
    for i in range(38):
        dic_[i] = [0]
    
    n_df = pd.DataFrame(data=dic_)
    n_df.columns = df.columns
    
    for cidade in df['codigo_ibge'].unique():
        n_df.loc[len(n_df)] = [2017, cidade] + df[df['codigo_ibge'] == cidade].iloc[:, 2:].mean().tolist()
    
    return n_df.drop(n_df.index[0])


def separacao(dados, cidades):
    df = dados[dados['codigo_ibge'] == cidades[0]]
    new_df_ = dados.drop(df.index)
    
    for cidade in cidades[1:]:
        df_ = dados[dados['codigo_ibge'] == cidade]
        new_df_ = new_df_.drop(df_.index)
        df = pd.concat([df,df_])
        
    return df, new_df_.reset_index().drop(columns=['index'])