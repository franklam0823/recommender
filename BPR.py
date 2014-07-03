# -*- coding: utf-8 -*-
import sys
import math
import random as random_built_in
import functools, operator
import collections
from numpy import *

from config import *
import f1

"""Assumption:

1.Suppose an item observed by user u represents user u likes it more than other which user u didn't meet.

2.Suppose an item's frequency of observation by user u determines the preference of user u toward the item.

3.Suppose an item observed by user u in recently is more valuabel than those items observed by user u several months ago or even more.

Xuij = (Xui - Xuj) * (8 - Imonth) ** -1 * -log(2, behavior_probability[Ibehavior])

"""


def sigmoid_func(x):
    return 1 / ( 1 + math.exp(-x) )


def R(U, V, B, vec, iJ, behavior_probability, num_item):
    user = vec[0]-1
    itemI = vec[1]-1
    itemJ = iJ
    behavior = vec[2]
    month = vec[3]
    frequency = vec[5]
    
    v1 = dot(U[user], V[itemI].T) + B[itemI] - dot(U[user], V[itemJ].T) + B[itemJ]
    v2 =  1 / (8-month)  
    v3 = 0.01/num_item * frequency
    v4 = 0.01/num_item * -(math.log(2, behavior_probability[behavior]))
    v = v1 * v2 + v3 + v4
    
    return v


def cal_distance(U, _U, V, _V, B, _B):
    num_user = U.shape[0]
    num_item = V.shape[0]

    U_test_line = random_built_in.randint(1, num_user-1)
    V_test_line = random_built_in.randint(1, num_item-1)
    B_test_line = random_built_in.randint(1, num_item-1)

    distance = (U[U_test_line] - _U[U_test_line]).sum() + (V[V_test_line] - _V[V_test_line]).sum() + (B[B_test_line] - _B[B_test_line]).sum()

    return distance


class BPR(object):

    def __init__(self):
        self.train_vec = loadtxt(raw_train_file_name, dtype=float) # (user, item, behavior, month, date)
        self.train_vec = self.train_vec[:, 0:5]
        self.test_vec = loadtxt(test_file_name, dtype=float) # (user, item, behavior, month, date)
        self.test_vec = self.test_vec[:, 0:5]

        self.num_train_data = len(self.train_vec)
        self.num_user = int(max(self.train_vec[:, 0].max(), self.test_vec[:, 0].max()))
        self.num_item = int(max(self.train_vec[:, 1].max(), self.test_vec[:, 1].max()))
        #self.R = spconvert(self.train_vec, self.num_user, self.num_item)

        beavior_vec = self.train_vec[:,2]
        statistic_s = collections.Counter(beavior_vec)

        self.Xuij = None
        self.data = []
        self.user_observed = {} #key is userid, value is a list of item observed by user

        self.DS = [] # (user, item i, item j, behavior, month, frequency)
        self.behavior_probability = {0:statistic_s[0.0]/self.num_train_data, 1:statistic_s[1.0]/self.num_train_data,
                                     2:statistic_s[2.0]/self.num_train_data, 3:statistic_s[3.0]/self.num_train_data}
        #print(self.behavior_probability)


    def preprocess_train_data(self):
        f = open(raw_train_file_name)
        
        temp = []
        [temp.append(d) for d in f]
        temp_dict = collections.Counter(temp)

        for t in temp_dict:
            temp = list(map(int,t.strip().split('\t')))
            temp.append(temp_dict[t])
            self.data.append(temp)

        self.data_size = len(self.data)
        self.data = array(self.data)

        f.close()
        for i in self.data:
            try:
                self.user_observed[i[0]].append(i[1])
            except:
                self.user_observed.update([(i[0], [i[1],])])


    def read_Ds_data_set(self):
        f = open(data_file_prefix + 'bpr_DS_data.txt')

        for d in f:
            self.DS.append(list(map(int,d.strip().split('\t'))))
        self.DS_size = len(self.DS)
        f.close


    def generate_Ds_data_set(self):

        f = open(data_file_prefix + 'bpr_DS_data.txt', 'w')
        for d in self.data:
            user = d[0] 
            item = d[1] 
            behavior = d[2]
            month = d[3] 
            date = d[4] 
            frequency = d[5] 
            random_item_sequence = random.permutation(self.num_item)[:50]

            s = ''
            for j in random_item_sequence:
                if j in self.user_observed[user]:
                    continue
                s = '\t'.join(list(map(str,[user, item, j, behavior, month, frequency])))
                f.write(s+'\n')

        f.close()
        

    def update_fun(self, U, V, B, vec, iJ, behavior_probability):
        alphaU = 0.01
        alphaV = 0.01

        betaV = 0.01
        gamma = 0.01

        user = vec[0] -1
        itemI = vec[1]-1
        itemJ = iJ
        behavior = vec[2]
        month = vec[3]
        frequency = vec[4]

        error_term = -sigmoid_func(-R(U, V, B, vec, itemJ, behavior_probability, self.num_item))

        U[user] = U[user] - gamma * (error_term * (V[itemI] - V[itemJ]) + alphaU * U[user])
        V[itemI] = V[itemI] - gamma * (error_term * U[user] + alphaV * V[itemI])
        V[itemJ] = V[itemJ] - gamma * (error_term * -U[user] + alphaV * V[itemJ])
        B[itemI] = B[itemI] - gamma * (error_term + betaV*B[itemI])
        B[itemJ] = B[itemJ] - gamma * (-error_term + betaV*B[itemJ])

        return U, V, B


    def bpr_func(self, k, max_epoch):

        U = random.rand(self.num_user, k) * 0.01
        V = random.rand(self.num_item, k) * 0.01
        B = random.rand(self.num_item, 1) * 0.01        
        
        train_vec = []
        
        max_batch = 1
        for t in range(max_batch):
            random_sequence = random.permutation(self.data_size)
            index = 0
            for di in range(self.data_size):
                i = self.data[di]
                user = i[0]
                itemI = i[1]
                itemJ = random_built_in.randint(0, self.num_item-1)
                while itemJ in self.user_observed[user]:
                    itemJ = random_built_in.randint(0, self.num_item-1)

                _U, _V, _B = U, V, B
                U, V, B = self.update_fun(U, V, B, i, itemJ, self.behavior_probability)

                distance = cal_distance(U, _U, V, _V, B, _B)
                #to do: check convergence


        P = dot(U, V.T)
        f = open(data_file_prefix + 'BPR_prediction.txt', 'w')
        PS = argsort(-P)   #a matrix of sorted arg of P

        current_user = 0
        for i in PS:
            u = i[:20]
            
            if u[0] == 3935:
                current_user += 1
                continue
            s = str(current_user) + '\t' + str(len(u)) + '\t'
            for j in u:
                s += str(j+1) + '\t'
            s += '\n'
            f.write(s)
            current_user += 1
        f.close()
        f1.main(data_file_prefix + 'BPR_prediction.txt')


if __name__ == '__main__':
    #generate_data.generate_train_data('user_item_feedback_month_date.txt_train')
    #generate_data.generate_test_data('user_item_feedback_month_date.txt_test')

    bpr = BPR()
    bpr.preprocess_train_data()
   
    #bpr.generate_Ds_data_set()
    #bpr.read_Ds_data_set()
    bpr.bpr_func(5, 100)
    

    
    


