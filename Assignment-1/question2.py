# Write a piece of code to obtain the least squares solution 
# $w_{ML}$ to the regression problem using the analytical solution.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata # for visualization
import seaborn as sns
sns.set_theme(style="whitegrid", palette="pastel")
import common
import question1

# gradient calculation
def calc_gradient(X, y, wt ):
  n = X.shape[0]
  gradient = 2 * np.dot(X.T, (np.dot(X, wt) - y))/n
  return gradient

# function to perform gradient descent and return weight history from each iteration
def gradient_descent(X: np.ndarray, y: np.ndarray, wt_initial: np.ndarray, learning_rate: float, iterations: int)->np.ndarray:

    no_of_observations, no_of_features = X.shape
    wt_curr = wt_initial
    wt_history = [wt_curr.copy()]

    for t in range(iterations):
        gradient = calc_gradient(X,y,wt_curr) #2 * np.dot(X.T, (np.dot(X, wt_curr) - y))/ no_of_observations  # Gradient of the cost function
        # print(f'iteration, t = {t}; gradient={gradient}')
        wt_curr = wt_curr - learning_rate * gradient  # Update the weights
        wt_history.append(wt_curr.copy())   # Store the weights at each step
        # print(wt_curr)

    return np.array(wt_history)

# function to compute the Euclidean norm ∥w_t − wML∥2
def norm_distance(wt_history, w_ml):
    wt_diffs = [np.linalg.norm(w - w_ml) for w in wt_history]
    return np.array(wt_diffs)

if __name__=="__main__":

    # load data
    X_train, y_train, X_test, y_test = common.get_data()

    ## Pick suitable $\eta$ : Lets check the plot $\|w_t - w_{ml}\|_2$ vs $t$ for different learning rate $\eta$
    # predict value using weights w_ml
    # Different learning rates to test
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5 ]
    # set parameters for gradient descent
    iterations = 100
    # get weights of  OLS
    wt_ml = question1.least_sq_sol(X_train, y_train)
    wt_initial = np.zeros(X_train.shape[1])  # Initialize weights to zeros
    # wt_initial = [np.random.rand() for i in range(X_train.shape[1])] # Initialize weights to random


    fig =plt.figure(0, figsize=(10, 6))
    plt.title('Gradient Descent: Norm Distance vs Iteration for Different Learning Rates')
    plt.xlabel('Iteration (t)')
    plt.ylabel(r'$\|w_t - w_{ML}\|_2$')

    # Perform gradient descent for each learning rate and plot the results
    for i, learning_rate in enumerate(learning_rates):
        # Run gradient descent
        wt_history = gradient_descent(X_train, y_train, wt_initial, learning_rate, iterations)

        # Compute the norm difference between wt and wt_ml at each iteration
        wt_diffs = norm_distance(wt_history, wt_ml)

        # Plot ∥wt − w_ml∥2 as a function of t (iteration number)
        plt.plot(wt_diffs,  label=f'Learning Rate = {learning_rate}')


    # Adjust layout
    plt.legend()
    plt.grid(True)
    fig.savefig("./figure_q2_01.png")
    # plt.show()

    ## run gradient descent for learning rate $\eta=0.1$ and collect the weight history
    # we can pick $\eta=0.25$ or $(0.1, 0.5)$ as the learning rate for gradient descent,  
    # since Euclidean norm distance approaches $0$ quickly compared to other $\eta$ values $<0.1$ tested above
    # Also 25 iterations is choosen as the error becomes close to zero

    # set parameters for gradient descent
    learning_rate = 0.25
    iterations = 25
    wt_initial = np.zeros(X_train.shape[1])  # Initialize weights to zeros
    # wt_initial = [np.random.rand() for i in range(X_train.shape[1])] # Initialize weights to random

    # run gradient descent
    wt_history = gradient_descent(X_train, y_train, wt_initial, learning_rate, iterations)

    print(f'weights for gradient descent= {wt_history[-1]}')

    # compute the norm difference between wt and wML at each iteration
    wt_diffs = norm_distance(wt_history, wt_ml)

    fig11 = plt.figure(11,figsize=(10, 6))
    # plot ∥wt − wML∥2 as a function of t (iteration number)
    plt.plot(wt_diffs, label=f'learning_rate,$\\eta$={learning_rate}')
    plt.xlabel('Iteration (t)')
    plt.ylabel(r'$\|w_t - w_{ml}\|_2$')
    plt.title(r'Norm difference $\|w_t - w_{ml}\|_2$ as a function of iteration $t$')
    plt.grid(True)
    plt.legend()
    fig11.savefig("./figure_q2_02.png")

    # predict value using weights w_ml
    y_pred_test = np.dot(X_test, wt_history[-1])

    y_pred_train = np.dot(X_train, wt_history[-1])

    # compute the MSE on the train set
    error_train = common.calculate_mse(y_train, y_pred_train)
    

    # compute the MSE on the test set
    error_test = common.calculate_mse(y_test, y_pred_test)
    

    print("======  solution results ===========")
    print(f'weights for least squares solution= {wt_ml}')
    print(f'Mean Squared Error (MSE) on train set(Gradient descent): {round(error_train,4)}')
    print(f'Mean Squared Error (MSE) on test set(Gradient descent): {round(error_test,4)}')


    plt.show()

