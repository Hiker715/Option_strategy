# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 15:51:29 2018

@author: Junhao
"""

import pandas as pd
import numpy as np
import arrow 
from itertools import product
import xgboost as xgb


def buy_high_c(x):
    try:
        return x.iloc[np.argmin(list(x['diff']))+1,]
    except:
        return x.iloc[np.argmin(list(x['diff'])),]
    
        
def buy_low_p(x):
    try:
        return x.iloc[np.argmin(list(x['diff']))-1,]
    except:
        return x.iloc[np.argmin(list(x['diff'])),]
    
    
def reg_fit(y,order,value):
    #y为一维数组类，order 为回归拟合的阶数，value为最终输出值
    #计算过去一段时间内的价格趋势，加速上涨为2，减速上涨为1， 加速下跌为 -2，减速下跌为-1
    try:
        y = np.array(y)
        if order == 2:
            x = np.arange(len(y))
            X = np.hstack((np.repeat(1,len(x)),x,x*x)).reshape(len(x),3,order='F')
            coef = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y)
            yhat = X.dot(coef)
            R2 = (yhat-y.mean()).dot((yhat-y.mean())) / (y-y.mean()).dot((y-y.mean()))
        if order == 1:
            x = np.arange(len(y))
            X = np.hstack((np.repeat(1,len(x)),x)).reshape(len(x),2,order='F')
            coef = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y)
            yhat = X.dot(coef)
            R2 = (yhat-y.mean()).dot((yhat-y.mean())) / (y-y.mean()).dot((y-y.mean()))
        if value == 'coef':
            return coef[-1]
        elif value == 'R2':
            return R2
        else: return np.nan
    except:
        return np.nan

def max_drawdown(data):
    Max_Drawdown = min(data*1.0/np.maximum.accumulate(data)-1)
    return Max_Drawdown


def performance(data,fee):
    signal = list(data['signal'])
    signal.insert(0,0)
    signal.append(0)
    total_fee = pd.Series(signal).diff().abs().sum()*fee
    result = {}
    result['Annualized_Return'] = data['strategy_return_NoFee'].mean()*250
    result['Annualized_Volatility'] = data['strategy_return_NoFee'].std()*np.sqrt(250)
    result['Sharpe'] = result['Annualized_Return'] / result['Annualized_Volatility']
    result['Max_Drawdown'] = max_drawdown((data['strategy_return_NoFee'] + 1).cumprod())
    result['Signal_1_Num'] = sum(data['signal']==1)
    result['Signal_-1_Num'] = sum(data['signal']==-1)
    result['Signal_Days_Rate'] = (result['Signal_1_Num']+result['Signal_-1_Num'])*1.0 / len(data)
    result['Annualized_Return_AfterFee'] = (data['strategy_return_NoFee'].sum()-total_fee)/len(data)*250
    result['Sharpe_AfterFee'] = result['Annualized_Return_AfterFee']/result['Annualized_Volatility']
    if result['Signal_1_Num'] != 0:
        result['win_ratio_day_1'] = sum(data[data['signal']==1]['strategy_return_NoFee']>0)*1.0/result['Signal_1_Num']
    else:
        result['win_ratio_day_1'] = 0
    if result['Signal_-1_Num'] != 0:
        result['win_ratio_day_-1'] = sum(data[data['signal']==-1]['strategy_return_NoFee']>0)*1.0/result['Signal_-1_Num']
    else:
        result['win_ratio_day_-1'] = 0
    if (result['Signal_1_Num']+result['Signal_-1_Num']) != 0:
        result['win_ratio_day'] = sum(data['strategy_return_NoFee']>0)*1.0/(result['Signal_1_Num']+result['Signal_-1_Num'])
    else:
        result['win_ratio_day'] = 0
    return pd.DataFrame(result,index=[0])


def get_xgb_feat_importances(clf):

    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()

    feat_importances = []
    for ft, score in fscore.iteritems():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    # Print the most important features and their importances
    #print feat_importances.head()
    return feat_importances


#input_dir = 'C:\Users\Junhao\Desktop'
#output_dir = 'C:\Users\Junhao\Desktop'
#output_name = 'option'

input_dir = '/Users/heisenberg/Downloads/研究生课件/1金融衍生品/期末'
output_dir = '/Users/heisenberg/Downloads/研究生课件/1金融衍生品/期末'
output_name = 'option'

options = pd.read_csv(r'%s\option.csv'%input_dir)
underlying = pd.read_csv(r'%s\underlying.csv'%input_dir)


###### Data Preprocessing
options['mid_price'] = (options.best_bid + options.best_offer)/2.0
options = pd.merge(options,underlying[['date','open']],on='date')
options['diff'] = np.abs(options.strike_price - options.open * 1000)
options = options.sort_values(by=['date','exdate','strike_price'])


options_c = options[options.cp_flag == 'C']
options_c['return'] = options_c.groupby(['exdate','strike_price'])['mid_price'].transform(lambda x:x.pct_change().shift(-1))
options_c_CurMon = options_c.groupby(['date']).apply(lambda x:x[x.exdate == np.min(x.exdate)])
options_c_NextMon = options_c.groupby(['date']).apply(lambda x:x[x.exdate == sorted(np.unique(x.exdate))[1]])

#当月最后一天的期权换成下月期权
options_c_CurMon = options_c_CurMon.groupby(['exdate']).apply(lambda x:x[x.date < np.unique(x.date)[-1]])
options_c_NextMon = options_c_NextMon.groupby(['exdate']).apply(lambda x:x[x.date >= np.unique(x.date)[-1]])
options_c_CurMon = pd.concat([options_c_CurMon,options_c_NextMon]).reset_index(drop=True).sort_values(by=['date','exdate','strike_price'])

#构建信号组合
options_c_CurMon_short2 = options_c_CurMon.groupby(['date'])[['return','mid_price','diff']].apply(lambda x: x.iloc[np.argmin(list(x['diff'])),]).reset_index()
options_c_CurMon_short2['return'] = options_c_CurMon_short2['return'] * (-1)
options_c_CurMon_short2['signal'] = -2
del options_c_CurMon_short2['diff']

options_c_CurMon_short1 = options_c_CurMon.groupby(['date'])[['return','mid_price','diff']].apply(buy_high_c).reset_index()
options_c_CurMon_short1['return'] = (options_c_CurMon_short1['return']*options_c_CurMon_short1['mid_price'] + options_c_CurMon_short2['return']*options_c_CurMon_short2['mid_price']) / (options_c_CurMon_short1['mid_price']+options_c_CurMon_short2['mid_price'])
options_c_CurMon_short1['signal'] = -1
del options_c_CurMon_short1['diff']


options_p = options[options.cp_flag == 'P']
options_p['return'] = options_p.groupby(['exdate','strike_price'])['mid_price'].transform(lambda x:x.pct_change().shift(-1))
options_p_CurMon = options_p.groupby(['date']).apply(lambda x:x[x.exdate == np.min(x.exdate)])
options_p_NextMon = options_p.groupby(['date']).apply(lambda x:x[x.exdate == sorted(np.unique(x.exdate))[1]])

#当月最后一天的期权换成下月期权
options_p_CurMon = options_p_CurMon.groupby(['exdate']).apply(lambda x:x[x.date < np.unique(x.date)[-1]])
options_p_NextMon = options_p_NextMon.groupby(['exdate']).apply(lambda x:x[x.date >= np.unique(x.date)[-1]])
options_p_CurMon = pd.concat([options_p_CurMon,options_p_NextMon]).reset_index(drop=True).sort_values(by=['date','exdate','strike_price'])

#构建信号组合
options_p_CurMon_long2 = options_p_CurMon.groupby(['date'])[['return','mid_price','diff']].apply(lambda x: x.iloc[np.argmin(list(x['diff'])),]).reset_index()
options_p_CurMon_long2['return'] = options_p_CurMon_long2['return'] * (-1)
options_p_CurMon_long2['signal'] = 2
del options_p_CurMon_long2['diff']

options_p_CurMon_long1 = options_p_CurMon.groupby(['date'])[['return','mid_price','diff']].apply(buy_low_p).reset_index()
options_p_CurMon_long1['return'] = (options_p_CurMon_long1['return']*options_p_CurMon_long1['mid_price'] + options_p_CurMon_long2['return']*options_p_CurMon_long2['mid_price']) / (options_p_CurMon_long1['mid_price']+options_p_CurMon_long2['mid_price'])
options_p_CurMon_long1['signal'] = 1
del options_p_CurMon_long1['diff']

return_signal_table = pd.concat([options_p_CurMon_long2,options_p_CurMon_long1,options_c_CurMon_short2,options_c_CurMon_short1]).sort_values(by=['date','signal'])

#### Underlying Timing Features


## Price Momentum

#过去i天的total return, i = 1, 3, 5, 10, 20, 60
for i in [1, 3, 5, 10, 20, 60, 120]:
    underlying['D_Acc_return_%d'%i] = (underlying['close'] - underlying['close'].shift(i)) * 1.0 / underlying['close'].shift(i)
del i

#计算D_Price_Change
for i,j in product([1, 3, 5, 10, 20, 60],[3, 5, 10, 20, 60, 120]):
    if i < j:
        underlying['D_Price_Change_%d_%d'%(i,j)] = underlying['close'].rolling(window=i).mean() / underlying['close'].rolling(window=j).mean()
del i,j

#计算D_Ret_Diff
for i,j in product([1, 3, 5, 10, 20, 60],[3, 5, 10, 20, 60, 120]):
    if i < j:
        underlying['D_Ret_Diff_%d_%d'%(i,j)] = underlying['D_Acc_return_1'].rolling(window=i).mean() - underlying['D_Acc_return_1'].rolling(window=j).mean()
del i,j

#计算price range and MA_price_range
underlying['D_PriceRange'] =  (underlying['high'] - underlying['low']) / underlying['close'].shift(1)
for i,j in product([1, 3, 5, 10, 20, 60],[3, 5, 10, 20, 60, 120]):
    if i < j:
        underlying['D_PriceRange_Change_%d_%d'%(i,j)] = underlying['D_PriceRange'].rolling(window=i).mean() - underlying['D_PriceRange'].rolling(window=j).mean()
del i,j

## Volatility Momentum

#以过去20天来计算每日波动率,然后计算 D_Vol_MA(i) - D_Vol_MA(j)
underlying['D_Vol'] = underlying['D_Acc_return_1'].rolling(window=20).std()
for i,j in product([1, 3, 5, 10, 20, 60],[3, 5, 10, 20, 60, 120]):
    if i < j:
        underlying['D_Vol_Change_%d_%d'%(i,j)] = underlying['D_Vol'].rolling(window=i).mean() - underlying['D_Vol'].rolling(window=j).mean()
del i,j

## Volume Momentum

#用每天的交易量计算Volume_MA(i)  - Volume_MA(j)
for i,j in product([1, 3, 5, 10, 20, 60],[3, 5, 10, 20, 60, 120]):
    if i < j:
        underlying['D_Volume_Change_%d_%d'%(i,j)] = underlying['volume'].rolling(window=i).mean() - underlying['volume'].rolling(window=j).mean()
del i,j

## Price Volume

#(Price_MA(i) - Price_MA(j))*(Volume_MA(i)-Volume_MA(j))
for i,j in product([1, 3, 5, 10, 20, 60],[3, 5, 10, 20, 60, 120]):
    if i < j:
        underlying['D_Price_Volume_Change_%d_%d'%(i,j)] = (underlying['close'].rolling(window=i).mean() - underlying['close'].rolling(window=j).mean()) * underlying['D_Volume_Change_%d_%d'%(i,j)]
del i,j

# MACD
underlying['D_MACD_Price'] = 2*((pd.ewma(underlying['close'],12) - pd.ewma(underlying['close'],26)) - pd.ewma((pd.ewma(underlying['close'],12) - pd.ewma(underlying['close'],26)),9))

underlying['D_MACD_Volume'] = 2*((pd.ewma(underlying['volume'],12) - pd.ewma(underlying['volume'],26)) - pd.ewma((pd.ewma(underlying['volume'],12) - pd.ewma(underlying['volume'],26)),9))

underlying['D_MACD_Range'] = 2*((pd.ewma(underlying['D_PriceRange'],12) - pd.ewma(underlying['D_PriceRange'],26)) - pd.ewma((pd.ewma(underlying['D_PriceRange'],12) - pd.ewma(underlying['D_PriceRange'],26)),9))

underlying['D_MACD_Vol'] = 2*((pd.ewma(underlying['D_Vol'],12) - pd.ewma(underlying['D_Vol'],26)) - pd.ewma((pd.ewma(underlying['D_Vol'],12) - pd.ewma(underlying['D_Vol'],26)),9))

#RSI

underlying['D_RSI_Price'] = (underlying['close'].diff() / underlying['close'].shift(1)).rolling(window=6).apply(lambda x: sum(x[x>0])*1.0/sum(np.abs(x))*100)

underlying['D_RSI_Volume'] = (underlying['volume'].diff() / underlying['volume'].shift(1)).rolling(window=6).apply(lambda x: sum(x[x>0])*1.0/sum(np.abs(x))*100)

underlying['D_RSI_Range'] = (underlying['D_PriceRange'].diff() / underlying['D_PriceRange'].shift(1)).rolling(window=6).apply(lambda x: sum(x[x>0])*1.0/sum(np.abs(x))*100)

underlying['D_RSI_Vol'] = (underlying['D_Vol'].diff() / underlying['D_Vol'].shift(1)).rolling(window=6).apply(lambda x: sum(x[x>0])*1.0/sum(np.abs(x))*100)

#ATR
for i in [14,30,60,125,250]:
    underlying['D_ATR_Price_%d'%i] = (pd.Series(np.max(pd.DataFrame([underlying['high']-underlying['low'],np.abs(underlying['high']-underlying['close'].shift(1)),np.abs(underlying['low']-underlying['close'].shift(1))]),axis=0))/underlying['close'].shift(1)).rolling(window=i).mean()
del i

for i,j in product([14,30,60,125],[30,60,125,250]):
    if i<j:
        underlying['D_ATR_Price_%d_%d'%(i,j)] = underlying['D_ATR_Price_%d'%i] - underlying['D_ATR_Price_%d'%j]
del i,j


## label and features

vol_window =30 # 用过去多少minute波动率对当期label进行修正

#reture of open to open
#underlying['return'] = underlying['open'].pct_change().shift(-2)

#return of open to close
underlying['return'] = ((underlying['close'] - underlying['open'])/underlying['open']).shift(-1)

#use vol to scale reture
#underlying['Vol'] = underlying['D_Acc_return_1'].rolling(window=vol_window).apply(lambda x:x.std())
underlying['Vol'] = 1

underlying['return/Vol']  = underlying['return'] / underlying['Vol']
#underlying = underlying[underlying['return/Vol'].notnull()]
underlying = underlying.reset_index(drop=True)

underlying['Y'] = underlying['date'].apply(lambda x:str(x)[:4])
underlying['YM'] = underlying['date'].apply(lambda x:str(x)[:6])
underlying = underlying[underlying['date']>20030101].reset_index(drop=True)

features_loc=pd.Series(underlying.columns).apply(lambda x :x[:2] == 'D_')
features = underlying[underlying.columns[features_loc]]


#### modeling

#使用rolling window 进行回测
RW = 600 #Rolling Window的宽度
Update_Period = 10 #模型更新的周期days
TradingDay_RW = np.unique(underlying['date'])
underlying['weekday'] = underlying['date'].apply(lambda x: arrow.get('%s'%x,'YYYYMMDD').isoweekday())

trade_out_list = pd.DataFrame()
trade_in_list = pd.DataFrame()
xgb_feat_importances = pd.DataFrame()
for i in range(RW,len(TradingDay_RW),Update_Period):
    in_sample_x = features[(underlying['date']<TradingDay_RW[i])&(underlying['date']>=TradingDay_RW[i-RW])]
    in_sample_y = underlying['return/Vol'][(underlying['date']<TradingDay_RW[i])&(underlying['date']>=TradingDay_RW[i-RW])]
    #in_sample_y[-1:] = np.array((underlying[underlying['date']==TradingDay_RW[i]]['close']-underlying[underlying['date']==TradingDay_RW[i]]['open'])/underlying[underlying['date']==TradingDay_RW[i]]['open'])/np.array(underlying[underlying['date']==TradingDay_RW[i-1]]['Vol'])
   
    if i < range(RW,len(TradingDay_RW),Update_Period)[-1]:
        out_sample_x = features[(underlying['date']>=TradingDay_RW[i])&(underlying['date']<TradingDay_RW[i+Update_Period])]
        out_sample_y = underlying['return/Vol'][(underlying['date']>=TradingDay_RW[i])&(underlying['date']<TradingDay_RW[i+Update_Period])]
    else:
        out_sample_x = features[underlying['date']>=TradingDay_RW[i]]
        out_sample_y = underlying['return/Vol'][underlying['date']>=TradingDay_RW[i]]

    dtrain = xgb.DMatrix(in_sample_x, label=in_sample_y)
    dtest = xgb.DMatrix(out_sample_x, label=out_sample_y)
        
    param = {
    'max_depth': 3,
    'objective': 'reg:linear',
    'eta': 0.05,
    'silent': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'scale_pos_weight': 1,
    'eval_metric': 'rmse',
    'nthread': 8,
    'seed': 0
    }
    num_round = 200; evals = [(dtrain, 'eval')]
    xgbmodel = xgb.train(param, dtrain, num_round, evals = evals)
    xgb_feat_importances_single = get_xgb_feat_importances(xgbmodel)
    xgb_feat_importances = pd.concat([xgb_feat_importances,xgb_feat_importances_single],ignore_index=True)
    if i < range(RW,len(TradingDay_RW),Update_Period)[-1]:
        trade_out_sample = underlying[(underlying['date']>=TradingDay_RW[i])&(underlying['date']<TradingDay_RW[i+Update_Period])][['date','weekday','YM','Y','return','Vol']]
    else:
        trade_out_sample = underlying[underlying['date']>=TradingDay_RW[i]][['date','weekday','YM','Y','return','Vol']]
    trade_out_sample_tree = pd.DataFrame()
    trade_in_sample = underlying[(underlying['date']<TradingDay_RW[i])&(underlying['date']>=TradingDay_RW[i-RW])][['date','weekday','YM','Y','return','Vol']]
    trade_in_sample_tree = pd.DataFrame()
    for ntree in [100,120,140,160,180,200]:
        trade_out_sample['predict'] = xgbmodel.predict(dtest, ntree_limit = ntree)
        trade_out_sample['predict'] = trade_out_sample['predict']*trade_out_sample['Vol']
        trade_out_sample['ntree'] = ntree
        trade_out_sample_tree = pd.concat([trade_out_sample_tree,trade_out_sample],ignore_index=True)
        trade_in_sample['predict'] = xgbmodel.predict(dtrain, ntree_limit = ntree)
        trade_in_sample['predict'] = trade_in_sample['predict']*trade_in_sample['Vol']
        trade_in_sample['ntree'] = ntree
        trade_in_sample_tree = pd.concat([trade_in_sample_tree,trade_in_sample],ignore_index=True)
    trade_out_list = pd.concat([trade_out_list,trade_out_sample_tree],ignore_index=True)
    trade_in_list = pd.concat([trade_in_list,trade_in_sample_tree],ignore_index=True)
xgb_feat_importances = xgb_feat_importances.groupby('Feature')['Importance'].mean().reset_index()
xgb_feat_importances = xgb_feat_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
#xgb_feat_importances.to_csv(r'%s\%s_%s_Features_Impormance.csv'%(output_dir,output_name,RW))
options_list = trade_out_list.copy()

del options_list['return']

#### 股票回测表现
#设置参数
signal_long_threshold = 0.003#信号阈值
signal_short_threshold = -0.003
fee = 0 #交易手续费

#trade_out_list = pd.read_csv(r'C:\Users\Junhao\Desktop\IC_Timing_RW30_EveryMinute_depth3_reg\IC_Rollingwindow200_depth3_signal0.001_fee0_reg_OutSample.csv')
#trade_in_list = pd.read_csv(r'C:\Users\Junhao\Desktop\IC_Timing_RW30_EveryMinute_depth3_reg\IC_Rollingwindow200_depth3_signal0.001_fee0_reg_InSample.csv')
trade_out_list['signal'] = np.where(trade_out_list['predict']>signal_long_threshold,1,np.where(trade_out_list['predict']<signal_short_threshold,-1,0))
trade_in_list['signal'] = np.where(trade_in_list['predict']>signal_long_threshold,1,np.where(trade_in_list['predict']<signal_short_threshold,-1,0))

trade_out_list['strategy_return_NoFee'] = trade_out_list['signal'] * trade_out_list['return']
trade_out_list.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_reg_OutSample.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold),index=False)

trade_in_list['strategy_return_NoFee'] = trade_in_list['signal'] * trade_in_list['return']
trade_in_list.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_reg_InSample.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold),index=False)

#根据ntree分类回测
results = trade_out_list.groupby(['ntree']).apply(performance,fee).reset_index()
del results['level_1']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),index=False)

results = trade_in_list.groupby(['ntree']).apply(performance,fee).reset_index()
del results['level_1']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_InSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),index=False)

#根据year进行回测
results = trade_out_list.groupby(['ntree','Y']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

results = trade_in_list.groupby(['ntree','Y']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_InSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

#根据weekday进行回测
results = trade_out_list.groupby(['ntree','weekday']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

results = trade_in_list.groupby(['ntree','weekday']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_InSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

#根据month进行回测
results = trade_out_list.groupby(['ntree','YM']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

results = trade_in_list.groupby(['ntree','YM']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_InSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

####期权回测表现
#设置参数
signal_long2_threshold = 0.005#信号阈值
signal_long1_threshold = 0.002
signal_short1_threshold = -0.002
signal_short2_threshold = -0.005
fee = 0 #交易手续费

options_list['signal'] = np.where(options_list['predict']>signal_long2_threshold,2,np.where(options_list['predict']>signal_long1_threshold,1,np.where(options_list['predict']<signal_short2_threshold,-2,np.where(options_list['predict']<signal_short1_threshold,-1,0))))
options_list['signal'] = options_list['signal'].shift(1)
options_list_result = pd.merge(options_list,return_signal_table,on=['date','signal'],how='left')
options_list_result['return'] = options_list_result['return'].fillna(0)
options_list_result['strategy_return_NoFee'] = options_list_result['return']
options_list_result.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_reg_OutSample.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold),index=False)


#根据ntree分类回测
results = options_list_result.groupby(['ntree']).apply(performance,fee).reset_index()
del results['level_1']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),index=False)

#根据year进行回测
results = options_list_result.groupby(['ntree','Y']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

#根据weekday进行回测
results = options_list_result.groupby(['ntree','weekday']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)

#根据month进行回测
results = options_list_result.groupby(['ntree','YM']).apply(performance,fee).reset_index()
del results['level_2']
results.to_csv(r'%s\%s_Daily_RW%sUP%s_signal%s_fee%s_reg_OutSample_performance.csv'%(output_dir,output_name,RW,Update_Period,signal_long_threshold,fee),mode='a',index=False)
