# -*- coding: utf-8 -*-
import sys
import math
import functools, operator
import collections
from numpy import *

from config import *
import f1, generate_data



def spconvert(vec, num_user, num_item):
    """return a full matrix with users' number plus one rows and items' number plus one columns"""
    rating = 0
    R = zeros((num_user, num_item))
    for lines in vec:
        user = lines[0]-1
        item = lines[1]-1
        rating = lines[2]
        if R[user][item] != 0:
            R[user][item] += rating    
        else:
            R[user][item] = rating
    return R


def cal_average(vec, num_user):
    """return a matrix with each rows average"""
    R = zeros((num_user, 1))
    index = 0
    for lines in vec :
        size = len([x for x in lines if x>0])
        R[index] = sum(lines) / (size + sys.float_info.epsilon)
        index += 1
    return R


def spconvert_avg( matrix, avg_matrix, num_user, num_item):
    """return a matrix with non-zero elements, replaced by average rating"""
    for i in range(num_user):
        for j in range(num_item):
            if matrix[i][j] == 0:
                matrix[i][j] = avg_matrix[i]
    return matrix


def PMF_updateU(train_vec, tradeoff_alpha, U, V):
    """Updata U for PMF"""

    num_user = U.shape[0]

    for u in range(num_user):
        #(user=u, item, rating)
        vec = train_vec[ train_vec[:, 0] == u+1]
    
        if vec.size != 0:
            # items' (rated by the current user) feature vectors
            vec_rows, vec_columns = vec.shape
            _V = V[vec[0][1],:]
            for v in range(1, vec_rows):
                _V = vstack((V[vec[v][1],:], _V))

            #if V is a vector
            if ndim(_V) == 1:
                b = sum(vec[:, 2].T * _V)
                A = dot(_V.T, _V) + tradeoff_alpha
                U[u,:] = b/A
            else:
                b = dot(vec[:, 2].T, _V)
                A = dot(_V.T, _V) + tradeoff_alpha * _V.shape[0] * eye(_V.shape[1])
                A = linalg.pinv(A)
                U[u,:] = dot(b , A)
    return U


def PMF_updateV(train_vec, tradeoff_alpha, U, V):
	"""USed for Update V"""

    num_item = V.shape[0]

    for i in range(num_item):
        #(user=u, item, rating)
        vec = train_vec[train_vec[:, 1] == i]
    
        if vec.size != 0:
            # user' (rate the current item) feature vectors
            vec_rows, vec_columns = vec.shape
            _U = U[vec[0][0],:]
            for u in range(1, vec_rows):
                _U = vstack((U[vec[u][0],:], _U))
            
            #if U is a vector
            if ndim(_U) == 1:
                b = sum(vec[:, 2].T * _U)
                A = dot(_U.T, _U) + tradeoff_alpha
                V[i,:] = b/A
            else:
                b = dot(vec[:, 2].T, _U)
                A = dot(_U.T, _U) + tradeoff_alpha * _U.shape[0] * eye(_U.shape[1])
                A = linalg.pinv(A)
                V[i,:] = dot(b , A)

    return V


def cal_rmse(U, V, valid_vec):
    rows, columns = valid_vec.shape
    sum_rmse = 0
    for p in range(rows):
        u = valid_vec[p][0]
        i = valid_vec[p][1]
        rp = dot(U[u], V[i].T)
        sum_rmse += ((rp - valid_vec[p][2]) ** 2)
    return math.sqrt(sum_rmse)


class Recommender(object):
    """Recommender class include pmf, hot, svd, hot_svd recommender algrithm"""

    def __init__(self): 
        self.train_vec = loadtxt(train_file_name, dtype=float) # (user, item, ratint/preference)
        self.train_vec = self.train_vec[:, 0:3]
        self.test_vec = loadtxt(test_file_name, dtype=float) # (user, item, behavior, month, date)
        self.test_vec = self.test_vec[:, 0:3]

        self.num_train_data = len(self.train_vec) #train data record number
        self.num_user = int(max(self.train_vec[:, 0].max(), self.test_vec[:, 0].max()))
        self.num_item = int(max(self.train_vec[:, 1].max(), self.test_vec[:, 1].max()))
        self.R = spconvert(self.train_vec, self.num_user, self.num_item) 


    def svd_func(self, k):
        """Standar SVD implementation"""        
        R = self.R
    
        user_average_rating = cal_average(R, self.num_user) 

        R = spconvert_avg(R, user_average_rating, self.num_user, self.num_item)

        rows, columns = R.shape
        for i in range(rows):
            R[i] = R[i] - user_average_rating[i]
       
        U, S, V = linalg.svd(R)
        U = U[:, 0:k]
        S = S[0:k]
        S = diag(S, 0)
        V = V[0:k,:]

        P = dot(U, S)
        P = dot(P, V)

        f = open(data_file_prefix + 'svd_prediction.txt', 'w')
        #print(P)
        PS = argsort(-P)   #a matrix of sorted arg of P
        #print(PS)

        current_user = 1
        for i in PS:
            u = i[:20]
            if int(u[1]) == 6351:
                current_user += 1
                continue
            s = str(current_user) + '\t' + str(len(u)) + '\t'
            for j in u:
                s += str(j+1) + '\t'
            s += '\n'
            f.write(s)
            current_user += 1
        f.close()
        f1.main(data_file_prefix + 'svd_prediction.txt')
        return PS


    def hot_func_person(self):
        """depend on user's history data to recommend"""
        user_predicion_dict = {}

        for i in range(self.num_user):
            predic_list = argsort(-self.R[i])[:10]
            user_predicion_dict.update([(i+1, predic_list)])

        f = open(data_file_prefix + 'hot_person_prediction.txt', 'w')
        for i in user_predicion_dict:
            s = str(i) + '\t' #+ str(len(user_predicion_dict[i])) + '\t'
            pred_item_num = 0
            pred_s = ''
            for j in user_predicion_dict[i]:
                # if j == 0: #because user i hasn't sufficient train data
                #     break
                pred_item_num += 1
                pred_s += str(j+1) + '\t'

            s = s + str(pred_item_num) + '\t' + pred_s + '\n'
            f.write(s)
        f.close()
        f1.main(data_file_prefix + 'hot_person_prediction.txt')

        return user_predicion_dict


    def svd_hot_hybrid_func(self, k):
        PS = self.svd_func(k)
        user_predicion_dict = self.hot_func_person()

        hybrid_predict = {}
        current_user = 0
        for i in PS:
            current_user += 1
            u = i[:20]
            try:
                hybrid_predict[current_user]
            except:
                hybrid_predict.update([(current_user, [])])
            for j in u:
                try:
                    if j in user_predicion_dict[current_user]:
                        hybrid_predict[current_user].append(j+1)
                except:
                    continue

        f = open(data_file_prefix + 'svd_hot_prediction.txt', 'w')
        for i in hybrid_predict:
            s = str(i) + '\t' 
            pred_item_num = 0
            pred_s = ''
            for j in hybrid_predict[i]:
                pred_item_num += 1
                pred_s += str(j) + '\t'

            s = s + str(pred_item_num) + '\t' + pred_s + '\n'
            f.write(s)
        f.close()
        f1.main(data_file_prefix + 'svd_hot_prediction.txt')
        

                
    def hot_func(self):
        most_popular_item = {}

        train_vec = self.train_vec[:, 1:]
        # print(len(train_vec))
        for l in train_vec:
            try:
                most_popular_item[l[0]] += l[1]
            except:
                most_popular_item.update([(l[0], l[1])])
        
        recommender_list = sorted(most_popular_item.items(), key=lambda s: s[1], reverse=True)

        f = open(data_file_prefix + 'hot_prediction.txt', 'w')
        for i in range(1, self.num_user+1):
            s = str(i) + '\t' + '20' + '\t'
            for j in range(20):
                s += str(int(recommender_list[j][0])) + '\t'
            s += '\n'
            f.write(s)
        f.close()
        f1.main(data_file_prefix + 'hot_prediction.txt')


    def pmf_func(self, k, epoch):
        max_epoch = epoch
        tradeoff_alpha = 0.01
        random_sequence = random.permutation(self.num_train_data)
        
        
        U = random.rand(self.num_user, k) * 0.01
        V = random.rand(self.num_item, k) * 0.01

        boundary = math.floor(self.num_train_data*0.8)

        train_vec = self.train_vec[:random_sequence[boundary], :] #used for training
        valid_vec = self.train_vec[random_sequence[boundary]+1:, :] #used for validation

        train_vec = self.train_vec[:boundary, :] #used for training
        valid_vec = self.train_vec[boundary+1:, :] #used for validation

        avg_rating = mean(train_vec[:, 2])
        train_vec[:, 2] -= avg_rating # (user, item , rating - avergeRating)

        avg_rating = mean(valid_vec[:, 2])
        valid_vec[:, 2] -= avg_rating # (user, item , rating - avergeRating)
        
        rmse = []
        max_batch = 1
    
        for t in range(max_epoch):
            #print(t)

            _U = U
            _V = V

            U = PMF_updateU(train_vec, tradeoff_alpha, _U, _V)
            V = PMF_updateV(train_vec, tradeoff_alpha, _U, _V)

            #check performance
            r = cal_rmse(U, V, valid_vec)
            #print(r)
            rmse.append(r)

            #check convergence
            if t > 0:
                if rmse[t-1] < rmse[t] or abs(rmse[t-1] - rmse[t]) < 0.0001:
                    break
                
        
        fu = open('pmf_u.txt', 'w')
        rowU, columnU = U.shape
        for i in range(rowU):
            temp = []
            s = ''
            for j in range(columnU):
                s += str(U[i][j]) + '\t' 
            s += '\n'
            fu.write(s)
        fu.close()

        fv = open('pmf_v.txt', 'w')
        rowV, columnV = V.shape
        for i in range(rowV):
            temp = []
            s = ''
            for j in range(columnV):
                s += str(V[i][j]) + '\t' 
            s += '\n'
            fv.write(s)
        fv.close()

        P = dot(U, V.T)
        f = open(data_file_prefix + 'pmf_prediction.txt', 'w')
        PS = argsort(-P)   #a matrix of sorted arg of P
        


        current_user = 1
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
        f1.main(data_file_prefix + 'pmf_prediction.txt')


if __name__ == '__main__':
    generate_data.generate_train_data('user_item_feedback_month_date.txt_train')
    generate_data.generate_test_data('user_item_feedback_month_date.txt_test')

    r = Recommender()
    #r.svd_func(60)
    #r.svd_hot_hybrid_func(10)
    
    r.hot_func_person()
    
    #r.hot_func()
    #r.pmf_func(60, 100)


