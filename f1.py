# -*- coding: utf-8 -*-

import sys
from config import *

"""
Compute the precision at k

This program compute the precision at k between prediction list and label list

k is the top k preditction in predicted list

"""

def pk(label, predict, k):
    """
    Compute the precision at k

    Parameters:
    label : dictionary, key is userid, value is items bought by user
            

    predict : dictionary , key is user id, value is items predicted to user
            

    k : int 
        the maximun number of predicted item
    """
    hit = 0
    #denominator = k * len(label)
    denominator = k * len(predict)
    #for ke in predict:
     #   denominator += len(predict[ke])

    i = 0
    for key in predict:
        i += 1
        pval = []
        lval = []
        try:
            lval = label[key]
        except:
            continue

        pval = predict[key]
        hit += len([val for val in lval if val in pval])
       
        #hit += len(list(set(pval) & set(lval)))

    print('Prediction@%d : %.4f' % (k, hit/(denominator+sys.float_info.epsilon)))
    return  hit/(denominator+sys.float_info.epsilon)


def rk(label, predict, k):

    hit = 0
    denominator = 0

    for ke in label:
        denominator += len(label[ke])
    i = 0
    for key in label:
        i+=1
        pval = []
        lval = []
        try:
            pval = predict[key]
        except:
            continue
        lval = label[key]
        hit += len([val for val in lval if val in pval])
    print('Recall@%d : %.4f' % (k, hit/(denominator+sys.float_info.epsilon)))    

    return  hit/(denominator+sys.float_info.epsilon)


def f1(p, r, k):
    f = 2*p*r/(p+r+sys.float_info.epsilon)
    print('F1@%d : %.4f\n' % (k, 2*p*r/(p+r+sys.float_info.epsilon)))    
    return f


def load(file_name):
    f = open(file_name)
    data = []
    [data.append(d.strip().split('\t')) for d in f]

    f.close()
    return data


def convert_dict(l, k):
    target = {}
    temp = []

    for p in l:
        [temp.append(i) for i in p]
        target.update([(p[0], temp[2:k+2])])
        temp = []

    return target



def main(prediction_file):

    predict = load(prediction_file)
    label = load(data_file_prefix + 'label.txt')
    #k = 2
    label_ini = label
    predict_ini = predict
    
    s = ''
    for k in range(1, 10):
        label = convert_dict(label_ini, 1000000)
        predict = convert_dict(predict_ini, k)
       # print(predict)
        pk_val = pk(label, predict, k)
        rk_val = rk(label, predict, k)
        res = f1(pk_val, rk_val, k)
        s += '@' + str(k) + '\t' + str(pk_val) + '\t' + str(rk_val) + '\t' + str(f1) + '\t' + '\n'
        
    return s



if __name__ == '__main__':
    
    prediction_file = ''
    main(prediction_file)

    
       

    
    
