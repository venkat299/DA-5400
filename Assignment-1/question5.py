# Assume that you would like to perform kernel regression on this dataset. 
# Which Kernel would you choose and why? 
# Code the Kernel regression algorithm and predict for the test data. 
# Argue why/why not the kernel you have chosen is a better kernel than the standard least squares regression.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata # for visualization
import seaborn as sns
sns.set_theme(style="whitegrid", palette="pastel")
import common
import question1
import question2

# kernel definitions

# degree n polynomial kernel function
def kernel_polynomial(x1, x2, degree=2, c=1):
    # Compute the dot product between every pair of points in x1 and x2
    # return (np.dot(x1, x2.T) + c) ** degree
    return ((x1 @ x2.T) + 1)**degree

# catenary kernel function
def kernel_catenary(x1, x2, a=1.0):
    # Compute the Euclidean distance between every pair of points in x1 and x2
    pairwise_dist = np.sqrt(((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2).sum(axis=2))
    # Compute the catenary kernel
    return np.cosh(pairwise_dist / a)



if __name__=="__main__":

    # load data
    X_train, y_train, X_test, y_test = common.get_data()

    # get weights of  OLS
    wt_ml = question1.least_sq_sol(X_train, y_train)

    n = X_train.shape[0]
    X_train2, X_eval, y_train2, y_eval = common.train_test_split(X_train, y_train, test_size=0.2, random_state=43)
    # Initialize the kernel matrix
    kernel_matrix = np.zeros((n, n))
    mse_kr=[]
     # for taking minimum degree
    polynomial_degree_list=[1,2,3,4]# np.arange(2,8,1)
    train_mse_list = []
    # Calculate the degree p kernel matrix by cross validation
    for p in polynomial_degree_list:
        km_train=((X_train2 @ X_train2.T) + 1)**p
        alpha=np.linalg.pinv(km_train )@ y_train2
        km=((X_train2 @ X_eval.T) + 1)**p
        y_pred=alpha@km
        train_mse_list.append(np.mean((y_pred - y_eval) ** 2))
    print(polynomial_degree_list)
    print(train_mse_list)


    min_index = np.argmin(train_mse_list)
    min_mse = train_mse_list[min_index]
    min_mse_degree= polynomial_degree_list[min_index]

    print(f'min training mse({min_mse:.4f}) occurs for degree={int(min_mse_degree)}')

    km_train=((X_train @ X_train.T) + 1)**2
    alpha2=np.linalg.pinv(km_train) @ y_train

    km_test=((X_train @ X_test.T) + 1)**2
    y_pred_test=alpha2@km_test


    km_train=((X_train @ X_train.T) + 1)**2
    y_pred_train=alpha2@km_train
    

    # compute the MSE on the train set
    error_train = common.calculate_mse(y_train, y_pred_train)
    
    # compute the MSE on the test set
    error_test = common.calculate_mse(y_test, y_pred_test)
    

    # visualize the feature and target
    # Creating a 3D plot to visualize the relationship between the two features and the target

    # Extract the two features and target from the data
    train_df = common.load_train_df()
    x1 = train_df.iloc[:, 0]
    x2 = train_df.iloc[:, 1]
    y = train_df.iloc[:, 2]

    # Create grid data for the contour plot
    grid_x, grid_y = np.mgrid[min(x1):max(x1):100j, min(x2):max(x2):100j]
    grid_z = griddata((x1, x2), y, (grid_x, grid_y), method='cubic')

   
    # Creating two 3D plots side by side, swapping the features in the second plot
    fig = plt.figure(0,figsize=(18, 7))

    # First plot: Feature 1 on x-axis, Feature 2 on y-axis
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(x1, x2, y, cmap='viridis', edgecolor='white',  linewidth=0.11, alpha=0.3)
    ax1.scatter3D(x1, x2, y_pred_train,)
    ax1.set_title('3D Plot: x1, x2, y ')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('y')

    # Second plot: rotate 90 degree
    ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot_trisurf(x2, x1, y, cmap='viridis', edgecolor='white', linewidth=0.11,)
    ax2.plot_trisurf(x1, x2, y, cmap='viridis', edgecolor='white',  linewidth=0.11, alpha=0.3)
    ax2.scatter3D(x1, x2, y_pred_train, )
    ax2.view_init(azim=-30, elev=45)
    ax2.set_title('3D Plot: x1, x2, y , rotated 30 degree')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('y')

    # plt.show()
    fig.savefig("./figure_q5_02.png")

    fig2 = plt.figure(1)
    # Plotting the predicted vs actual values
    plt.scatter(y_test, y_pred_test)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Kernel Regression: Predicted vs Actual')
    fig2.savefig("./figure_q5_03.png")

    plt.show()

    print(f'Mean Squared Error (MSE) on train set: {round(error_train,4)}')
    print(f'Mean Squared Error (MSE) on test set: {round(error_test,4)}')
