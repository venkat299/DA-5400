# Code the gradient descent algorithm for ridge regression. 
# Cross-validate for various choices of $λ$ and plot the error in the validation set as a function of λ. 
# For the best $λ$ chosen, obtain $w_R$. 
# Compare the test error (for the test data in the file FMLA1Q1Data test.csv) of $w_R$ with $w_{ML}$. 
# Which is better and why?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata # for visualization
import seaborn as sns
sns.set_theme(style="whitegrid", palette="pastel")
import common
import question1
import question2

def calc_ridge_gradient(X, y, wt, _lambda):
  n = X.shape[0]
  gradient =  2 * np.dot(X.T, (np.dot(X, wt) - y))/ n  + 2 * _lambda * wt
  return gradient


def ridge_regression_gradient_descent(X, y, _lambda, learning_rate, num_iterations):
    N, D = X.shape
    wt = np.zeros(D)  # Initialize weights
    for i in range(num_iterations):
        gradient = calc_ridge_gradient(X, y, wt, _lambda)
        wt = wt - learning_rate * gradient
    return wt

def kfold_cross_validation(X, y, lambdas, k=5, learning_rate=0.01, num_iterations=1000):
    fold_size = len(X) // k
    validation_errors = []

    for lambda_ in lambdas:
        fold_errors = []
        for fold in range(k):
            # Create train and validation sets
            _X_val = X[fold * fold_size: (fold + 1) * fold_size]
            _y_val = y[fold * fold_size: (fold + 1) * fold_size]

            _X_train = np.concatenate((X[:fold * fold_size], X[(fold + 1) * fold_size:]), axis=0)
            _y_train = np.concatenate((y[:fold * fold_size], y[(fold + 1) * fold_size:]), axis=0)

            # Train Ridge Regression
            _wt_ridge = ridge_regression_gradient_descent(_X_train, _y_train, lambda_, learning_rate, num_iterations)

            _y_pred = _X_val.dot(_wt_ridge)

            # Compute validation error (MSE)
            fold_errors.append(common.calculate_mse(_y_pred, _y_val))


        # print(fold_errors, np.median(fold_errors))
        # Average validation error across folds
        validation_errors.append(np.median(fold_errors))

    return validation_errors

class LR_Ridge:
  def __init__(self, _lambda=0):
    self._lambda = _lambda

  def train(self,X, y ):
    self.wt = ridge_regression_gradient_descent(X_train, y_train, self._lambda, learning_rate, num_iterations)
    return self

  def predict(self, X_test):
    return  X_test.dot(self.wt)

  def mse(self, X, y):
    return common.calculate_mse(self.predict(X), y)

  def params(self):
    return {"lambda" : self._lambda, "weights":self.wt}



if __name__=="__main__":

    # load data
    X_train, y_train, X_test, y_test = common.get_data()

    # get weights of  OLS
    wt_ml = question1.least_sq_sol(X_train, y_train)

    # choose lambda

    # Parameters for Ridge Regression
    lambdas = np.linspace(0,0.5, 20)

    # print(lambdas)
    learning_rate = 0.01
    num_iterations = 1000

    # print(X_train.shape)

    # Cross-validation for Ridge Regression
    validation_errors = kfold_cross_validation(X_train, y_train, lambdas, k=5, learning_rate=learning_rate, num_iterations=num_iterations)

    # ============
    # print(validation_errors)
    idx = validation_errors.index(min(validation_errors))
    min_lambda = lambdas[idx]
    

    model_ridge = LR_Ridge(_lambda = min_lambda).train(X_train, y_train)

    train_mse = model_ridge.mse(X_train, y_train)
    test_mse = model_ridge.mse(X_test, y_test)


    print(f'train_mse = {train_mse}')
    print(f'test_mse = {test_mse}')
    # ============


    # Plot validation error as a function of lambda
    fig0 = plt.figure(0, figsize=(10, 6))
    plt.plot(lambdas, validation_errors, marker='o')
    # plt.yscale('log')
    # plt.yscale('log')
    # Add a small margin to y-limits (10% margin)
    y_min = min(validation_errors)
    y_max = max(validation_errors)
    margin = 0.1 * (y_max - y_min)
    x_min = min(lambdas)
    x_max = max(lambdas)
    margin = 0.1 * (x_max - x_min)
    plt.xlim([x_min - margin, x_max + margin])

    # Find the index of the minimum y-value
    min_index = np.argmin(validation_errors)
    x_min = lambdas[min_index]
    y_min = validation_errors[min_index]


    wt_r = model_ridge.params()["weights"]

    plt.xlabel('Lambda (Regularization Parameter)')
    plt.ylabel('Mean Validation Error (MSE) in k fold CV')
    plt.title('Mean Validation Error vs Lambda for Ridge Regression (k-fold)')

    plt.scatter(x_min, y_min, color='red', zorder=5, label=f"Min Point (x={x_min:.2f}, y={y_min:.2f})")

    plt.annotate(f"Min (lambda={x_min:.4f}, MSE={y_min:.2f})",
                 xy=(x_min, y_min),
                 xytext=(x_min, y_min+2),
                 arrowprops=dict(arrowstyle="->", color='red'))

    plt.grid(True)
    fig0.savefig("./figure_q4_01.png")


    y_pred_train = model_ridge.predict(X_train)
    y_pred_test = model_ridge.predict(X_test)

    # compute the MSE on the train set
    error_train = common.calculate_mse(y_train, y_pred_train)
    
    # compute the MSE on the test set
    error_test = common.calculate_mse(y_test, y_pred_test)
    


    print("====== Analytical solution results ===========")
    print(f'min validation error = {min(validation_errors)} occurs at lambda = {min_lambda}')
    print(f'weights for Ridge Regression = {wt_r}')
    print(f'Mean Squared Error (MSE) for Ridge Regression on train set: {round(error_train,4)}')
    print(f'Mean Squared Error (MSE) on Ridge Regression  test set: {round(error_test,4)}')


    # analyse the effect of regularisation on the features
    alphas = lambdas = np.linspace(0,10, 200)
    coefs = []
    errors_coefs = []

    w_ml = [9.89400832,   1.76570568,   3.5215898]
    # Train the model with different regularisation strengths
    for a in alphas:
        m = LR_Ridge(_lambda = a).train(X_train, y_train)
        coefs.append(m.params()["weights"])
        errors_coefs.append(common.calculate_mse(m.params()["weights"], w_ml))


    alphas = pd.Index(alphas, name="lambda $\\lambda$")
    coefs = pd.DataFrame(coefs, index=alphas, columns=["bias", "x1", "x2"])
    errors = pd.Series(errors_coefs, index=alphas, name="Mean squared error")

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    coefs.plot(
        ax=axs[0],
        logx=True,
        title="Ridge coefficients as a function of the regularization strength",
    )
    axs[0].set_ylabel("Ridge coefficient values")
    errors.plot(
        ax=axs[1],
        logx=True,
        title="Coefficient error as a function of the regularization strength",
    )
    _ = axs[1].set_ylabel("Mean squared error")
    fig.savefig("./figure_q4_02.png")

    plt.show()




