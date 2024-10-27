# Code the stochastic gradient descent algorithm using 
# batch size of 100 and plot  ∥wt−wML∥2  as a function of t. 
# What are your observations?


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata # for visualization
import seaborn as sns
sns.set_theme(style="whitegrid", palette="pastel")
import common
import question1
import question2


if __name__=="__main__":

    # load data
    X_train, y_train, X_test, y_test = common.get_data()

    np.random.seed(13)
    # Parameters for SGD
    batch_size = 100
    learning_rate = 0.11
    num_iterations = 400

    # get weights of  OLS
    wt_ml = question1.least_sq_sol(X_train, y_train)

    # Initialize weights for SGD
    wt = np.zeros(X_train.shape[1])
    wt_avg = wt
    norm_distances = []

    # SGD loop
    for t in range(num_iterations):
        # Sample a random batch
        indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
        X_batch, y_batch = X_train[indices], y_train[indices]

        # Compute the gradient for the current batch
        gradient = question2.calc_gradient(X_batch, y_batch, wt)

        # Update weights
        wt = wt - learning_rate * gradient
        wt_avg = ((wt_avg*t)+wt)/(t+1)
        wt = wt_avg

        # Compute the norm distance from the optimal solution
        norm_distances.append(np.linalg.norm(wt - wt_ml))

    print(f'weights for stochastic gradient = {wt}')
    # predict value using weights w_ml
    y_pred_test = np.dot(X_test, wt)

    y_pred_train = np.dot(X_train, wt)

    # compute the MSE on the train set
    error_train = common.calculate_mse(y_train, y_pred_train)
    

    # compute the MSE on the test set
    error_test = common.calculate_mse(y_test, y_pred_test)
    

    print("======  solution results ===========")
    print(f'weights for stochastic gradient solution= {wt}')
    print(f'Mean Squared Error (MSE) on train set(stochastic gradient descent): {round(error_train,4)}')
    print(f'Mean Squared Error (MSE) on test set(stochastic gradient descent): {round(error_test,4)}')

    # Plot the results
    fig = plt.figure(0,figsize=(10, 6))
    plt.plot(norm_distances)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|w_t - w_{ML}\|_2$')
    plt.title('Stochastic Gradient Descent: $|w_t - w_{ML}|_2$ vs Iterations')
    plt.grid(True)
    fig.savefig("./figure_q3_01.png")
    plt.show()



#     Observation
# The norm  ∥wt−wML∥2  decreases over time, indicating that the SGD algorithm is converging toward the optimal solution  wML .

# Unlike traditional gradient descent, stochastic gradient descent exhibits fluctuations due to the random nature of sampling batches. However, it still converges in the long run.