import pickle
import numpy as np
import matplotlib.pyplot as plt

# with open('/home/kamakshi/Desktop/hw1/data1_py2.pkl', 'rb') as f:
    #  data = pickle.load(f)
def Covariance(data,j):
	path = str(data)
	#     file = HW1_data/'+str(data)+'.pkl'
	with open(path, 'rb') as f:
		data = pickle.load(f)

	X = np.asarray(data)
	cov_matrix = np.cov(X.T) #transposed to convert data from nx2 to 2xn matrix
	print('The Covariance Matrix is = ')
	print(cov_matrix)
	print("\n")
	plt.figure(1)
	plt.suptitle('HW1 part1 for data'+str(j+1), fontsize=20 )
	title = 'HW1 Part 1'+'data'+str(j)
	plt.title(title)
	plt.subplot(2,2,1)
	plt.title('Original data with Eigen Vectors')
	plt.scatter(X[:,0],X[:,1]) # since not transposed its a nx2 matrix

	mean = X-np.mean(X,axis=0)  #axis=0 is mean for individual rows
	eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
	print('Eigen Values for data'+str(j+1)+':')
	print(eig_vals)
	print('\n')
	print('Eigen Vectors for data'+str(j+1)+':')
	print(eig_vecs)
	print("\n")
	origin = mean
	x = eig_vecs[0,:]
	y = eig_vecs[1,:]
	plt.quiver( x, y, color=['r','b'],scale=2)


	#scaling matrix
	sx= 0.5
	sy=0.4
	Scale = np.array([[sx, 0], [0, sy]])

	#applying matrix
	Scaled_X = X.dot(Scale)

	Scaled_Cov =  np.cov(Scaled_X.T)
	# print("Scaled_Cov \n",Scaled_Cov)

	#rotation matrix
	theta = (np.pi)/3
	cos, sin = np.cos(theta), np.sin(theta)
	Rotation = np.array([[cos, -sin], [sin, cos]])
	# Transformation matrix
	T = Scale.dot(Rotation)
	# print("Transformation matrix after Rotation n scaling",T)
	# print("\n")

	# Apply transformation matrix to X
	D = X.dot(T)
	# print("data after transformation \n", D)
	plt.subplot(2,2,2)
	plt.title('Transformed data vs Original data')
	plt.scatter(X[:, 0], X[:, 1],color='red',label='old data')
	plt.scatter(D[:, 0], D[:, 1],color='blue',label='transformed data')
	plt.legend()


	C_Mt= np.cov(D.T)
	eig_Vat, eig_Vect = np.linalg.eig(C_Mt)

	plt.subplot(2,2,3)
	plt.title('Transformed data with Eigen Vectors')
	plt.scatter(D[:, 0], D[:, 1])
	for s, r in zip(eig_Vat, eig_Vect.T):
		plt.plot([0, 3*np.sqrt(s)*r[0]], [0, 3*np.sqrt(s)*r[1]], 'r-', lw=3)



	#uncorrelated data
	#eigen decomposition
	C = np.cov(D.T)

	# Calculate eigenvalues
	eVa, eVe = np.linalg.eig(C)

	# Calculate transformation matrix from eigen decomposition
	R, S = eVe, np.diag(np.sqrt(eVa))
	T = R.dot(S).T

	# Transform data with inverse transformation matrix T^-1
	Z = D.dot(np.linalg.inv(T))
	plt.subplot(2,2,4)
	plt.scatter(Z[:, 0], Z[:, 1])
	plt.title('Uncorrelated Data')
	plt.axis('equal')
	plt.show()

	# Covariance matrix of the uncorrelated data
	white=np.cov(Z.T)
	print('Covariance Matrix for uncorrelated data'+str(j+1)+':')
	print(white)
	print("\n")
	print('##############################################################################')

def main():

	for j in range(3):

		data = './data'+ str(j+1) +'_py2.pkl'
		Covariance(data,j)

if __name__== "__main__":
    main()
