import argparse
import os, sys
import pickle
import numpy as np
import random as rnd

# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass


import matplotlib.pyplot as plt
###Regularization
def ls_reg(x, y,cov_matrix):

    X = np.vstack([x, np.ones(len(x))]).T
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    l = min(eig_vals[0],eig_vals[1])
    l = eig_vals[0]/eig_vals[1]
    print(l)
    lamda= np.diag([l,l])                             #Tuning parameter for regularization
    a=np.linalg.inv((X.T).dot(X)+lamda)
    return (a.dot(X.T)).dot(y)





def matrix_lstsqr(x, y):
    # Computes the least-squares solution to a linear matrix equation.
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

def RANSAC(data,j,e):
    path = str(data)
    #     file = HW1_data/'+str(data)+'.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)

    data = np.asarray(data)
    x = data[:,0]
    y = data[:,1]
    # x = np.asarray(x)
    # y = np.asarray(y)

    ds = len(data)
    print(ds)
    # t = np.sqrt(3.84*np.std([x.T,y.T])**2)
    # t = 15 # np.sqrt(3.84*np.std([x.T,y.T])**2)
    t=18
    print(t)
    s = 2                                             #Min number of points to fit a line
    p = 0.99                                          #Probability for inliers

    N = int(round(np.log(1-p)/np.log(1-(1-e)**s)))    #Number of times the loop is run
    print(N)

    # inliers = np.ones(200)

    for i in range(N) :
        inliers=[]
        outliers=[]
        pts = rnd.sample(range(0,ds-1),s)
        # print(pts,x[pts])
        slope, intercept = matrix_lstsqr(x[pts], y[pts])
        line_x = [round(min(x)) - 5, round(max(x)) + 5]
        line_y = [slope*x_i + intercept for x_i in line_x]
        # print(slope)
        a = -slope/(slope**2 + 1)**0.5
        b = 1/(slope**2 + 1)**0.5
        # print(a,b)
        er = np.absolute(a*x + b*y - intercept)
        inliers = [er <= t]
        outliers = [er > t]
        # plt.scatter(x[inliers],y[inliers],c='red')
        # plt.scatter(x[outliers],y[outliers],c='blue')
        # plt.plot(line_x, line_y, color='red')
        # plt.show()
        # for k in range(ds):
        #     er = np.absolute(a*x[k] + b*y[k] - intercept)
        #     inliers[k] = (er < t)
        # print(inliers)
        # print('outliers',x[outliers])
        print('length of outliers:',len(x[outliers]))
        Z = float(len(x[outliers]))
        print(type(Z))
        if ((Z/ds) < e) :
            print([(Z/ds)<e])
            slope, intercept = matrix_lstsqr(x[inliers], y[inliers])

            #line_t1 = [slope*x_i + intercept+(t/slope) for x_i in line_x]
            #line_t2 = [slope*x_i + intercept-(t/slope) for x_i in line_x]

            a = -slope/(slope**2 + 1)**0.5
            b = 1/(slope**2 + 1)**0.5

            line_x = [round(min(x)) - 5, round(max(x)) + 5]
            # line_y = [-a*x_i/b + (intercept)/b for x_i in line_x]
            line_y = [slope*x_i + intercept for x_i in line_x]
            line_t1 = [-a*x_i/b + (intercept+t)/b for x_i in line_x]
            line_t2 = [-a*x_i/b + (intercept-t)/b for x_i in line_x]

            plt.figure(2)
            plt.suptitle('HW1 part3 for data'+str(j+1), fontsize=20 )
            l = 'Part 3'+' data'+str(j)
            plt.title(l)
            plt.subplot(2,2,j+1)
            cov_matrix = np.cov(data.T)
            slope_2,intercept_2= ls_reg(x, y,cov_matrix)
            # print('regularized',slope_2,intercept_2)
            line_w = [round(min(x)) - 1, round(max(x)) + 1]
            line_z = [slope_2*x_i + intercept_2 for x_i in line_w]
            plt.plot(line_w, line_z, color='purple',label='regularization')


            plt.scatter(x[inliers],y[inliers],c='red',label='inliers')
            plt.scatter(x[outliers],y[outliers],c='blue',label='outliers')
            plt.plot(line_x, line_y, color='black',label='best fit line')
            plt.plot(line_x, line_t1, color='green',linestyle='dashed',label='threshold')
            plt.plot(line_x, line_t2, color='green',linestyle='dashed')
            plt.legend()
            break

    # plt.show()

def main():

    for j in range(3):
        #data = '/home/kamakshi/Desktop/hw1/data'+ str(j+1) +'_py2.pkl'
        data = './data'+ str(j+1) +'_py2.pkl'
        # data_plot(data,i)
        # plt.show()
        if j == 0:
            e=0.1
        elif j ==1:
            e = 0.5
        elif j==2:
            e = 0.7
        RANSAC(data,j,e)
    plt.show()


if __name__== "__main__":
    main()
