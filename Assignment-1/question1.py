# Write a piece of code to obtain the least squares solution 
# $w_{ML}$ to the regression problem using the analytical solution.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata # for visualization
import seaborn as sns
sns.set_theme(style="whitegrid", palette="pastel")

import common


def least_sq_sol(X:np.ndarray, y:np.ndarray) -> np.ndarray:
    # analytical solution wt_ml = (X^T X)^(-1) X^T y
    X_T_X_inv = np.linalg.inv(np.dot(X.T, X))  # Inverse of (X^T X)
    X_T_y = np.dot(X.T, y)                     # X^T y
    wt_ml = np.dot(X_T_X_inv, X_T_y)           # wt_ml = (X^T X)^(-1) X^T y

    return wt_ml


if __name__=="__main__":

    # load data
    X_train, y_train, X_test, y_test = common.get_data()

    # calculate the weights
    wt_ml = least_sq_sol(X_train, y_train)

    # print("The least squares solution (wt_ml) is:", wt_ml)
    

    # predict value using weights w_ml
    y_pred_train = np.dot(X_train, wt_ml)
    y_pred_test = np.dot(X_test, wt_ml)


    # compute the MSE on the train set
    error_train = common.calculate_mse(y_train, y_pred_train)
    
    # compute the MSE on the test set
    error_test = common.calculate_mse(y_test, y_pred_test)
    


    print("====== Analytical solution results ===========")
    print(f'weights for least squares solution= {wt_ml}')
    print(f'Mean Squared Error (MSE) on train set: {round(error_train,4)}')
    print(f'Mean Squared Error (MSE) on test set: {round(error_test,4)}')


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
    fig.savefig("./figure_q1_02.png")

    fig2 = plt.figure(1)
    # Plotting the predicted vs actual values
    plt.scatter(y_test, y_pred_test)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('OLS solution: Predicted vs Actual')
    fig2.savefig("./figure_q1_03.png")
    plt.show()

