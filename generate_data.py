# -*- coding: utf-8 -*-

import math
import functools, operator
import collections
from numpy import *

from config import *


def generate_train_data(filename):
    """Create a file with a set of (user, item, rating/preference) """

    train_vec = loadtxt(raw_train_file_name, dtype=float) # (user, item, behavior, month, date)
    train_vec = train_vec[:, 0:5]
    test_vec = loadtxt(test_file_name, dtype=float) # (user, item, behavior, month, date)
    test_vec = test_vec[:, 0:5]

    num_train_data = len(train_vec)
    num_user = int(max(train_vec[:, 0].max(), test_vec[:, 0].max()))
    num_item = int(max(train_vec[:, 1].max(), test_vec[:, 1].max()))



    beavior_vec = train_vec[:,2]
    statistic_s = collections.Counter(beavior_vec)

    tbeavior_vec = test_vec[:,2]
    tstatistic_s = collections.Counter(tbeavior_vec)
   
    behavior_probability = {0:statistic_s[0.0]/num_train_data, 1:statistic_s[1.0]/num_train_data,
                            2:statistic_s[2.0]/num_train_data, 3:statistic_s[3.0]/num_train_data}

    temp = []
    data = []
    f = open(data_file_prefix + filename)
    [temp.append(d) for d in f] 
    f.close()
    temp_dict = collections.Counter(temp) 

    for t in temp_dict:
        temp = list(map(int,t.strip().split(',')))
        #temp[2] = behavior_dict[temp[2]]
        temp[2] = -math.log(2,behavior_probability[temp[2]])
        temp[3] = (1/(PREDICT_MOTH-temp[3]))
        temp[4] = math.log(temp_dict[t])  #frequency
        data.append(temp)


    f = open(data_file_prefix + 'train_set.txt', 'w')
    ratings = []
    for d in data:
        rating = d[2] * d[3] + d[4]
        ratings.append(rating)
        s = str(d[0]) + '\t' + str(d[1]) + '\t' + str(rating) + '\n'
        f.write(s)
    f.close()


def generate_test_data(filename):
    """create a file with a set of date (userid, number of item bought by user, list of item bought by user)"""
    test_set = []
    purchase_dict = {}
    
    f = open(data_file_prefix + filename)
    [test_set.append(d.strip('\n').split(',')) for d in f]    
    f.close()

    t_set = []

    for d in test_set:
        if d[2] == '1':
            t_set.append(d)
    list(map(lambda d: purchase_dict.update([(d[0], ())]), t_set))

    for d in t_set:
        purchase_dict[d[0]] += (d[1],)

    f = open(data_file_prefix + 'label.txt', 'w')

    for i in purchase_dict:
        s = str(i) + '\t' + str(len(purchase_dict[i])) + '\t' + '\t'.join(purchase_dict[i])  + '\n'
        f.write(s)
    f.close()
    


if __name__ == '__main__':    
    generate_train_data('user_item_feedback_month_date.txt_train')
    generate_test_data('user_item_feedback_month_date.txt_test')

