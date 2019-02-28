#! /usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt

##### Part 2-HW1
def matrix_lstsqr(x, y):
    # Computes the least-squares solution to a linear matrix equation.
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

def ls_reg(x, y,cov_matrix):

    X = np.vstack([x, np.ones(len(x))]).T
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    l = min(eig_vals[0],eig_vals[1])
    # l = eig_vals[0]/eig_vals[1]
    # print(l)
    lamda= np.diag([l,l])                             #Tuning parameter for regularization
    a=np.linalg.inv((X.T).dot(X)+lamda)
    return (a.dot(X.T)).dot(y)

def check(u):
    index =0
    if u[0]>= u[1]:
        index=1
    else:
        index=0
    return index

def data_plot(data,i):
    path = str(data)
    #     file = HW1_data/'+str(data)+'.pkl'
    with open(path, 'rb') as f:
    	data = pickle.load(f)
    # Computes the least-squares solution to a linear matrix equation.
    array = np.asarray(data)
    x = array[:,0]
    y = array[:,1]
    slope,intercept= matrix_lstsqr(x, y)
    print('LS',slope,intercept)
    # print("slope and intercept for data"+str(i+1),slope,intercept)
    line_x = [round(min(x)) - 1, round(max(x)) + 1]
    line_y = [slope*x_i + intercept for x_i in line_x]

    # orthogonal distance
    cov_matrix = np.cov(array.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    eig_min = min(eig_vals)
    index = check(eig_vals)
    a = eig_vecs[0,index]
    b = eig_vecs[1,index]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    d = a*mean_x + b*mean_y
    line_p = [round(min(x)) - 1, round(max(x)) + 1]
    line_q = [(d-(a*x_i))/b for x_i in line_p]


    #Least Squares Fitting with Regularization
    #print("parameters for ls with regularization is \n",slope_2,intercept_2 )
    slope_2,intercept_2= ls_reg(x, y,cov_matrix)
    print('regularized',slope_2,intercept_2)
    line_w = [round(min(x)) - 1, round(max(x)) + 1]
    line_z = [slope_2*x_i + intercept_2 for x_i in line_w]


    #plots
    plt.figure(1)
    plt.suptitle('HW1 part2', fontsize=20 )
    # plt.title('Part 2')
    plt.subplot(2,2,i+1)
    plt.title('Plot for data'+str(i+1))
    plt.scatter(x,y)
    plt.plot(line_x, line_y, color='red', label='vertical')
    # ftext = 'y = ax + b = {:.3f} + {:.3f}x \n'\
            # .format(slope, intercept)
    # plt.figtext(.15,.8, ftext, fontsize=11, ha='left')
    plt.plot(line_p, line_q, color='yellow',label='orthogonal')

    plt.plot(line_w, line_z, color='black',label='regularization')
    plt.legend()

#######################################################################################################################################
def main():

    for i in range(3):
        data = './data'+ str(i+1) +'_py2.pkl'
        data_plot(data,i)
    plt.show()



if __name__== "__main__":
    main()
