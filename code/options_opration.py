#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 04:38:49 2018

@author: heisenberg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:52:31 2018

@author: heisenberg
"""

import pandas as pd
import numpy as np


option = pd.read_csv('/Users/heisenberg/Downloads/研究生课件/1金融衍生品/期末/option.csv')
under = pd.read_csv('/Users/heisenberg/Downloads/研究生课件/1金融衍生品/期末/underlying.csv')
underp = pd.read_csv('/Users/heisenberg/Downloads/研究生课件/1金融衍生品/期末/option_Daily_RW600UP10_signal0.003_reg_OutSample的副本.csv')

option.drop(['last_date', 'volume', 'open_interest', 'impl_volatility', 'delta',
       'gamma', 'vega', 'theta', 'cfadj', 'ss_flag', 'index_flag',
       'issue_type', 'issuer', 'exercise_style'], axis=1, inplace=True)
under.drop(['index_flag', 'exchange_d', 'class', 'issue_type',  'cfadj'], axis=1, inplace=True)
underp.drop(['return', 'mid_price', 'strategy_return_NoFee'], axis=1, inplace=True)

option = option.loc[option.best_bid!=0,:]
option = pd.merge(option, under[['date','open']], on='date')
option['diff'] = np.abs(option.strike_price - option.open * 1000)

aunder = underp.loc[underp.ntree==200]
aunder = underp
aunder.drop(['weekday', 'YM', 'Y', 'Vol', 'ntree'], axis=1, inplace=True)

# 
# 五个操作函数，输出当前情况最佳操作
def signal_2(tmp_date, option = option):
    # 返回最佳买入看涨期权
    # 买入看涨期权，价格小于开盘价，而且最接近开盘价的期权
    # 当天所有可买看涨期权
    # open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='C') & (option.date==tmp_date)]
    # 然后选出执行月份最接近且价格最接近的期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    op = tmp_option.iloc[0, :]
    return (op.exdate, op.cp_flag, op.strike_price, op.best_offer, 1)

#def signal_1(tmp_date, option = option):
#    # 牛市价差，依次返回最佳买入看涨期权和最佳卖出看涨期权
#    # 买入低执行价格看涨期权，卖空高执行价格的看涨期权，买入的看涨期权（价格小于开盘价，而且最接近开盘价的实值期权）
#    open_p = option[option.date==tmp_date].open.unique()[0]
#    tmp_b_option = option[(option.cp_flag=='C') & (option.date==tmp_date) & (option.strike_price<open_p*1000)]
#    tmp_s_option = option[(option.cp_flag=='C') & (option.date==tmp_date) & (option.strike_price>=open_p*1000)]
#    # 然后选出执行月份最接近且价格最高的实值看涨期权
#    tmp_b_option = tmp_b_option.sort_values(by=['exdate','strike_price'], ascending=[True, False])
#    # 选出执行月份最接近，且价格最低的虚值看涨期权
#    tmp_s_option = tmp_s_option.sort_values(by=['exdate','strike_price'], ascending=[True, True])
#    # 最佳买入和最佳卖出
#    op_b = tmp_b_option.iloc[0, :]
#    op_s = tmp_s_option.iloc[0, :]
#    return (op_b.exdate, op_b.cp_flag, op_b.strike_price, op_b.best_offer, 1), (op_s.exdate, op_s.cp_flag, op_s.strike_price, op_s.best_bid, -1)

def signal_1(tmp_date, option = option):
    # 牛市价差，依次返回最佳买入看涨期权和最佳卖出看涨期权
    # 买入低执行价格看涨期权，卖空高执行价格的看涨期权，买入的看涨期权（价格小于开盘价，而且最接近开盘价的实值期权）
    # open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='C') & (option.date==tmp_date)]
    # 然后选出执行月份最接近且价格最高的实值看涨期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    # 选出执行月份最接近，且价格最低的虚值看涨期权
    # 最佳买入和最佳卖出
    op_b = tmp_option.iloc[1, :]
    op_s = tmp_option.iloc[0, :]
    if op_b.strike_price > op_s.strike_price:
        tmp = op_b
        op_b = op_s
        op_s = tmp
    else:
        pass
    return (op_b.exdate, op_b.cp_flag, op_b.strike_price, op_b.best_offer, 1), (op_s.exdate, op_s.cp_flag, op_s.strike_price, op_s.best_bid, -1)



def signal_n_1(tmp_date, option = option):
    # 熊市价差，依次返回最佳买入看跌期权和最佳卖出看跌期权
    # 买入高执行价格看跌期权，卖空低执行价格的看跌期权，买入的看跌期权（价格大于开盘价，而且最接近开盘价的实值期权）
    # open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='P') & (option.date==tmp_date)]
    # 然后选出执行月份最接近且价格最低的实值看跌期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    # 选出执行月份最接近，且价格最高的虚值看跌期权
    # 最佳买入和最佳卖出
    op_b = tmp_option.iloc[1, :]
    op_s = tmp_option.iloc[0, :]
    if op_b.strike_price < op_s.strike_price:
        tmp = op_b
        op_b = op_s
        op_s = tmp
    else:
        pass
    return (op_b.exdate, op_b.cp_flag, op_b.strike_price, op_b.best_offer, 1), (op_s.exdate, op_s.cp_flag, op_s.strike_price, op_s.best_bid, -1)


def signal_n_2(tmp_date, option = option):
    # 返回最佳买入看跌期权
    # 买入看跌期权，价格高于开盘价，而且最接近开盘价的实值期权
    # 当天所有可买看跌期权
    #  open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='P') & (option.date==tmp_date)]
    # 然后选出执行月份最接近且价格最低的期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    op = tmp_option.iloc[0, :]
    return (op.exdate, op.cp_flag, op.strike_price, op.best_offer, 1)
    
# 特殊情况，到期日前一天，交易下个月的期权，不买本月的了，而且要把手中的全部平仓
    
# 五个操作函数进行相应修改 
def signal_2_last(tmp_date, option = option):
    # 返回最佳买入看涨期权
    # 买入看涨期权，价格小于开盘价，而且最接近开盘价的实值期权
    # 当天所有可买看涨期权
    # open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='C') & (option.date==tmp_date)]
    # 然后选出执行月份"次"接近且价格最高的期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    # 选出月份过程
    exdate_ = sorted(tmp_option.exdate.unique())
    sec_exdate = exdate_[1]
    tmp_option = tmp_option[tmp_option.exdate==sec_exdate]
    # 接原函数
    op = tmp_option.iloc[0, :]
    return (op.exdate, op.cp_flag, op.strike_price, op.best_offer, 1)

def signal_1_last(tmp_date, option = option):
    # 牛市价差，依次返回最佳买入看涨期权和最佳卖出看涨期权
    # 买入低执行价格看涨期权，卖空高执行价格的看涨期权，买入的看涨期权（价格小于开盘价，而且最接近开盘价的实值期权）
    # open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='C') & (option.date==tmp_date)]
    # 然后选出执行月份最接近且价格最高的实值看涨期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    # 选出月份过程
    exdate_ = sorted(tmp_option.exdate.unique())
    sec_exdate = exdate_[1]
    tmp_option = tmp_option[tmp_option.exdate==sec_exdate]
    # 接原函数
    # 最佳买入和最佳卖出
    op_b = tmp_option.iloc[1, :]
    op_s = tmp_option.iloc[0, :]
    if op_b.strike_price > op_s.strike_price:
        tmp = op_b
        op_b = op_s
        op_s = tmp
    else:
        pass
    return (op_b.exdate, op_b.cp_flag, op_b.strike_price, op_b.best_offer, 1), (op_s.exdate, op_s.cp_flag, op_s.strike_price, op_s.best_bid, -1)

def signal_n_1_last(tmp_date, option = option):
    # 熊市价差，依次返回最佳买入看跌期权和最佳卖出看跌期权
    # 买入高执行价格看跌期权，卖空低执行价格的看跌期权，买入的看跌期权（价格大于开盘价，而且最接近开盘价的实值期权）
    # open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='P') & (option.date==tmp_date)]
    # 然后选出执行月份最接近且价格最低的实值看跌期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    # 选出月份过程
    exdate_ = sorted(tmp_option.exdate.unique())
    sec_exdate = exdate_[1]
    tmp_option = tmp_option[tmp_option.exdate==sec_exdate]
    # 接原函数
    # 最佳买入和最佳卖出
    try:
        op_b = tmp_option.iloc[1, :]
        op_s = tmp_option.iloc[0, :]
        if op_b.strike_price < op_s.strike_price:
            tmp = op_b
            op_b = op_s
            op_s = tmp
        else:
            pass
    except:
        tmp_b_option = option[(option.cp_flag=='P') & (option.date==tmp_date) & (option.strike_price>open_p*1000)]
        tmp_s_option = option[(option.cp_flag=='P') & (option.date==tmp_date) & (option.strike_price<=open_p*1000)]
        # 然后选出执行月份最接近且价格最低的实值看跌期权
        tmp_b_option = tmp_b_option.sort_values(by=['exdate','strike_price'], ascending=[True, True])
        # 选出执行月份最接近，且价格最高的虚值看跌期权
        tmp_s_option = tmp_s_option.sort_values(by=['exdate','strike_price'], ascending=[True, False])
        # 最佳买入和最佳卖出
        op_b = tmp_b_option.iloc[0, :]
        op_s = tmp_s_option.iloc[0, :]
        print('n_1_last_wrong')
    return (op_b.exdate, op_b.cp_flag, op_b.strike_price, op_b.best_offer, 1), (op_s.exdate, op_s.cp_flag, op_s.strike_price, op_s.best_bid, -1)


def signal_n_2_last(tmp_date, option = option):
    # 返回最佳买入看跌期权
    # 买入看跌期权，价格高于开盘价，而且最接近开盘价的实值期权
    # 当天所有可买看跌期权
    # open_p = option[option.date==tmp_date].open.unique()[0]
    tmp_option = option[(option.cp_flag=='P') & (option.date==tmp_date)]
    # 然后选出执行月份最接近且价格最低的期权
    tmp_option = tmp_option.sort_values(by=['exdate','diff'], ascending=[True, True])
    # 选出月份过程
    exdate_ = sorted(tmp_option.exdate.unique())
    sec_exdate = exdate_[1]
    tmp_option = tmp_option[tmp_option.exdate==sec_exdate]
    # 接原函数
    op = tmp_option.iloc[0, :]
    return (op.exdate, op.cp_flag, op.strike_price, op.best_offer, 1)

def buy_signal_opration(tmp_date, hold, tmp_op_b, option, deal_cnt, return_):
    if len(hold) == 0:
        hold.add(tmp_op_b)
        deal_cnt += 1
    elif len(hold) == 1:
        if tmp_op_b in hold:
            pass
        else:
            # 手里的期权不是需要的期权，卖掉手里的，计算收益率，同时买进需要的
            # 还需要查到手里的期权到价格
            tmp_hold = list(hold)[0]
            try:
                if tmp_hold[-1] == 1:
                    try:
                        tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_bid'].values[0]
                        
                        return_ = return_*((tmp_price - tmp_hold[-2])/(tmp_hold[-2]*3)+1)
                        hold.remove(tmp_hold)
                    except:
                        hold.remove(tmp_hold)
                    hold.add(tmp_op_b)
                    # 双向操作，交易次数加2
                    deal_cnt += 2

                else:
                    try:
                        tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_offer'].values[0]
                        return_ = return_*((tmp_hold[-2] - tmp_price)/(tmp_hold[-2]*3)+1)
                        hold.remove(tmp_hold)
                    except:
                        hold.remove(tmp_hold)
                    hold.add(tmp_op_b)
                    # 双向操作，交易次数加2
                    deal_cnt += 2
            except:
                print('find opsite deal wrong')
    else:
        if len(hold) == 2:
            if tmp_op_b in hold:
                pass
            else:
                # 手里有两个期权，但和手中的不一样，卖掉手中的，买进需要的
                tmp_hold = list(hold)
                # 判断手中两个期权是是long还是short，分别做操作
                if tmp_hold[0][-1] == 1:
                    try:
                        tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0][0]) & (option.cp_flag==tmp_hold[0][1]) & (option.strike_price==tmp_hold[0][2]), 'best_bid'].values[0]
                        return_ = return_*((tmp_price - tmp_hold[0][-2])/(tmp_hold[0][-2]*3)+1)
                        hold.remove(tmp_hold[0])
                    except:
                        hold.remove(tmp_hold[0])
                    deal_cnt += 1
                elif tmp_hold[0][-1] == -1:
                    try:
                        tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0][0]) & (option.cp_flag==tmp_hold[0][1]) & (option.strike_price==tmp_hold[0][2]), 'best_offer'].values[0]
                        return_ = return_*((tmp_hold[0][-2] - tmp_price)/(tmp_hold[0][-2]*3)+1)
                        hold.remove(tmp_hold[0])
                    except:
                        hold.remove(tmp_hold[0])
                    deal_cnt += 1
                elif tmp_hold[1][-1] == 1:
                    try:
                        tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[1][0]) & (option.cp_flag==tmp_hold[1][1]) & (option.strike_price==tmp_hold[1][2]), 'best_bid'].values[0]
                        return_ = return_*((tmp_price - tmp_hold[1][-2])/(tmp_hold[1][-2]*3)+1)
                        hold.remove(tmp_hold[1])
                    except:
                        hold.remove(tmp_hold[1])
                    deal_cnt += 1
                elif tmp_hold[1][-1] == -1:
                    try:
                        tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[1][0]) & (option.cp_flag==tmp_hold[1][1]) & (option.strike_price==tmp_hold[1][2]), 'best_offer'].values[0]
                        return_ = return_*((tmp_hold[1][-2] - tmp_price)/(tmp_hold[1][-2]*3)+1)
                        hold.remove(tmp_hold[1])
                    except:
                        hold.remove(tmp_hold[1])
                else:
                    print('buy_opration_wrong')
                hold.add(tmp_op_b)
                deal_cnt += 1
                
                
        else:
            print('buy_opration_wrong')
    
    return hold, deal_cnt, return_


def buy_sell_signal_operation(tmp_date, hold, tmp_op_b, tmp_op_s, option, deal_cnt, return_):
    # 需要做一个双向操作
    if len(hold) == 0:
        hold.add(tmp_op_b)
        hold.add(tmp_op_s)
        deal_cnt += 2
    elif len(hold) == 1:
        if tmp_op_b in hold:
            # 有一个期权在，只需要在交易另外一个
            hold.add(tmp_op_s)
            deal_cnt += 1
        elif tmp_op_s in hold:
            # 有一个期权在，只需要在交易另外一个
            hold.add(tmp_op_b)
        else:
            # 两个期权都不在，平仓已有的期权，计算收益率，交易需要的价差组合（加进hold即可）
            tmp_hold = list(hold)[0]
            try:
                if tmp_hold[-1] == 1:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_bid'].values[0]
                    
                    return_ = return_*((tmp_price - tmp_hold[-2])/(tmp_hold[-2]*3)+1)
                    hold.remove(tmp_hold)
                    hold.add(tmp_op_b)
                    hold.add(tmp_op_s)
                    # 双向操作，交易次数加2
                    deal_cnt += 3

                else:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_offer'].values[0]
                    return_ = return_*((tmp_hold[-2] - tmp_price)/(tmp_hold[-2]*3)+1)
                    hold.remove(tmp_hold)
                    hold.add(tmp_op_b)
                    hold.add(tmp_op_s)
                    # 双向操作，交易次数加2
                    deal_cnt += 3
            except:
                print('wrong')
                
    else:
        if len(hold) == 2:
            # 最复杂的情况，手中有两个，还要交易两个
            # 先判断，把需要平仓的平掉，再加上需要的仓位
            hold_b = list(hold)[0]
            hold_s = list(hold)[1]
            if hold_b[-1] ==1:
                pass
            else:
                tmp = hold_b
                hold_b = hold_s
                hold_s = tmp
            if tmp_op_b in hold and tmp_op_s in hold:
                pass
            elif tmp_op_b in hold and tmp_op_s not in hold:
                # 把手中的hold_s平掉，再卖空一个tmp_op_s
                try:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==hold_s[0]) & (option.cp_flag==hold_s[1]) & (option.strike_price==hold_s[2]), 'best_offer'].values[0]
                    return_ = return_*((hold_s[-2] - tmp_price)/(hold_s[-2]*3)+1)
                    hold.remove(hold_s)
                except:
                    hold.remove(hold_s)
                hold.add(tmp_op_s)
                deal_cnt += 2
            elif tmp_op_b not in hold and tmp_op_s in hold:
                # 把手中的hold_b平掉，再卖空一个tmp_op_b
                try:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==hold_b[0]) & (option.cp_flag==hold_b[1]) & (option.strike_price==hold_b[2]), 'best_bid'].values[0]
                    return_ = return_*((tmp_price - hold_b[-2])/(hold_b[-2]*3)+1)
                    hold.remove(hold_b)
                except:
                    hold.remove(hold_b)
                hold.add(tmp_op_b)
                deal_cnt += 2
            else:
                # 平仓持有的，加上需要的仓位
                try:
                    tmp_s_price = option.loc[(option.date==tmp_date) & (option.exdate==hold_s[0]) & (option.cp_flag==hold_s[1]) & (option.strike_price==hold_s[2]), 'best_offer'].values[0]
                    return_ = return_*((hold_s[-2] - tmp_s_price)/(hold_s[-2]*3)+1)
                    hold.remove(hold_s)
                except:
                    hold.remove(hold_s)
                try:
                    tmp_b_price = option.loc[(option.date==tmp_date) & (option.exdate==hold_b[0]) & (option.cp_flag==hold_b[1]) & (option.strike_price==hold_b[2]), 'best_bid'].values[0]
                    return_ = return_*((tmp_b_price - hold_b[-2])/(hold_b[-2]*3)+1)
                    hold.remove(hold_b)
                except:
                    hold.remove(hold_b)
                hold.add(tmp_op_b)
                hold.add(tmp_op_s)
                deal_cnt += 4
        else:
            print('buy_sell_operation_wrong')
            
    return hold, deal_cnt, return_

def close_position(tmp_date, hold, option, deal_cnt, return_):
    # 平仓操作
    if len(hold) == 0:
        pass
    elif len(hold) == 1:
        tmp_hold = list(hold)[0]
        try:
            if tmp_hold[-1] == 1:
                tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_bid'].values[0]
                return_ = return_*((tmp_price - tmp_hold[-2])/tmp_hold[-2]+1)
                hold.remove(tmp_hold)
                deal_cnt += 1

            else:
                tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_offer'].values[0]
                return_ = return_*((tmp_hold[-2] - tmp_price)/tmp_hold[-2]+1)
                hold.remove(tmp_hold)
                deal_cnt += 1
        except:
            print('find opsite deal wrong')
    else:
        if len(hold) == 2:
            tmp_hold = list(hold)[0]
            try:
                if tmp_hold[-1] == 1:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_bid'].values[0]
                    return_ = return_*((tmp_price - tmp_hold[-2])/tmp_hold[-2]+1)
                    hold.remove(tmp_hold)
                    deal_cnt += 1
        
                else:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_offer'].values[0]
                    return_ = return_*((tmp_hold[-2] - tmp_price)/tmp_hold[-2]+1)
                    hold.remove(tmp_hold)
                    deal_cnt += 1
            except:
                print('find opsite deal wrong')
            tmp_hold = list(hold)[1]
            try:
                if tmp_hold[-1] == 1:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_bid'].values[0]
                    return_ = return_*((tmp_price - tmp_hold[-2])/tmp_hold[-2]+1)
                    hold.remove(tmp_hold)
                    deal_cnt += 1
        
                else:
                    tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_offer'].values[0]
                    return_ = return_*((tmp_hold[-2] - tmp_price)/tmp_hold[-2]+1)
                    hold.remove(tmp_hold)
                    deal_cnt += 1
            except:
                print('find opsite deal wrong')
        else:
            print('close_position_wrong')
    
    return hold, deal_cnt, return_
        
        
    

# 交易如何操作
# 计算累计收益率，返回收益率和交易次数
# option_df:包含开盘价
# under_df:包含日期和当天的信号
# hold表示手中持有的情况，集合，元素为元组，形式如：（exdate，cp_flag, strike_price, buy/sell price, long/short）
all_exdate = sorted(option.exdate.unique())
all_exdate = set([edate-1 for edate in all_exdate])
all_exdate.update([edate-2 for edate in all_exdate])
all_exdate.update([edate-3 for edate in all_exdate])
all_exdate.update([edate-4 for edate in all_exdate])

hold = set()
deal_cnt = 0
return_ = 1


# 现在设定交易规则
# 手里有看涨空头或多头，需要交易看跌，平仓看涨；
# 手里有看涨空头或多头，需要交易看涨，检查手中的仓位，再做操作
# 手里有看跌空头或多头，需要交易看涨，平仓看跌；
# 手里有看跌空头或多头，需要交易看跌，检查手中仓位，再做操作
# 查看手中的仓位情况
# 如果是最后一天需要所有平仓
for date in list(aunder.date)[:10]:
    tmp_date = date
    tmp_signal = int(aunder.loc[aunder.date==tmp_date, 'signal'])
    tmp_op_b = ()
    tmp_op_s = ()
    if tmp_signal == 0:
        continue
    else:
        if tmp_date not in all_exdate:
            if tmp_signal == 2:
                tmp_op_b = signal_2(tmp_date, option)
                hold, deal_cnt, return_ = buy_signal_opration(tmp_date, hold, tmp_op_b, option, deal_cnt, return_)
                            
            elif tmp_signal == 1:
                tmp_op_b, tmp_op_s = signal_1(tmp_date, option)
                hold, deal_cnt, return_ = buy_sell_signal_operation(tmp_date, hold, tmp_op_b, tmp_op_s, option, deal_cnt, return_)
                        
            elif tmp_signal == -1:
                tmp_op_b, tmp_op_s = signal_n_1(tmp_date, option)
                # 需要一个双向操作
                hold, deal_cnt, return_ = buy_sell_signal_operation(tmp_date, hold, tmp_op_b, tmp_op_s, option, deal_cnt, return_)
                
            elif tmp_signal == -2:
                tmp_op_b = signal_n_2(tmp_date, option)
                # 买入看跌期权
                hold, deal_cnt, return_ = buy_signal_opration(tmp_date, hold, tmp_op_b, option, deal_cnt, return_)
        #if tmp_date in all_exdate:
        else:
            if tmp_signal == 2:
                tmp_op_b = signal_2_last(tmp_date, option)
                # 类似于signal==2
                hold, deal_cnt, return_ = buy_signal_opration(tmp_date, hold, tmp_op_b, option, deal_cnt, return_)
            elif tmp_signal == 1:
                tmp_op_b, tmp_op_s = signal_1_last(tmp_date, option)
                hold, deal_cnt, return_ = buy_sell_signal_operation(tmp_date, hold, tmp_op_b, tmp_op_s, option, deal_cnt, return_)
                
            elif tmp_signal == -1:
                tmp_op_b, tmp_op_s = signal_n_1_last(tmp_date, option)
                hold, deal_cnt, return_ = buy_sell_signal_operation(tmp_date, hold, tmp_op_b, tmp_op_s, option, deal_cnt, return_)
                
            elif tmp_signal == -2:
                tmp_op_b = signal_n_2_last(tmp_date, option)
                hold, deal_cnt, return_ = buy_signal_opration(tmp_date, hold, tmp_op_b, option, deal_cnt, return_)
    
hold, deal_cnt, return_ = close_position(20140530, hold, option, deal_cnt, return_)
            




# 单向操作，替换函数前的
if len(hold) == 0:
    hold.add(tmp_op_b)
    deal_cnt += 1
elif len(hold) == 1:
    if tmp_op_b in hold:
        #continue
        pass
    else:
        # 手里的期权不是需要的期权，卖掉手里的，计算收益率，同时买进需要的
        # 还需要查到手里的期权到价格
        tmp_hold = list(hold)[0]
        try:
            if tmp_hold[-1] == 1:
                tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_bid']
                return_ = return_*((tmp_price - tmp_hold[-2])/tmp_hold[-2]+1)
                hold.remove(tmp_hold)
                hold.add(tmp_op_b)
                # 双向操作，交易次数加2
                deal_cnt += 2

            else:
                tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0]) & (option.cp_flag==tmp_hold[1]) & (option.strike_price==tmp_hold[2]), 'best_offer']
                return_ = return_*((tmp_hold[-2] - tmp_price)/tmp_hold[-2]+1)
                hold.remove(tmp_hold)
                hold.add(tmp_op_b)
                # 双向操作，交易次数加2
                deal_cnt += 2
        except:
            print('find opsite deal wrong')
elif len(hold) == 2:
    if tmp_op_b in hold:
        #continue
        pass
    else:
        # 手里有两个期权，但和手中的不一样，卖掉手中的，买进需要的
        tmp_hold = list(hold)
        # 判断手中两个期权是是long还是short，分别做操作
        if tmp_hold[0][-1] == 1:
            tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0][0]) & (option.cp_flag==tmp_hold[0][1]) & (option.strike_price==tmp_hold[0][2]), 'best_bid']
            return_ = return_*((tmp_price - tmp_hold[0][-2])/tmp_hold[0][-2]+1)
            hold.remove(tmp_hold[0])
            deal_cnt += 1
        elif tmp_hold[0][-1] == -1:
            tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[0][0]) & (option.cp_flag==tmp_hold[0][1]) & (option.strike_price==tmp_hold[0][2]), 'best_offer']
            return_ = return_*((tmp_hold[0][-2] - tmp_price)/tmp_hold[0][-2]+1)
            hold.remove(tmp_hold[0])
            deal_cnt += 1
        elif tmp_hold[1][-1] == 1:
            tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[1][0]) & (option.cp_flag==tmp_hold[1][1]) & (option.strike_price==tmp_hold[1][2]), 'best_bid']
            return_ = return_*((tmp_price - tmp_hold[1][-2])/tmp_hold[1][-2]+1)
            hold.remove(tmp_hold[1])
            deal_cnt += 1
        elif tmp_hold[1][-1] == -1:
            tmp_price = option.loc[(option.date==tmp_date) & (option.exdate==tmp_hold[1][0]) & (option.cp_flag==tmp_hold[1][1]) & (option.strike_price==tmp_hold[1][2]), 'best_offer']
            return_ = return_*((tmp_hold[1][-2] - tmp_price)/tmp_hold[1][-2]+1)
            hold.remove(tmp_hold[1])
        hold.add(tmp_op_b)
        deal_cnt += 1




