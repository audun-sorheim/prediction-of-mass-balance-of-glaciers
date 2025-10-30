import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# import the training and validation and test dataset
def import_datasets(loc):

    train_data = pd.read_pickle(r'data_NA\NA_' + loc + 'train_data.pkl')
    val_data = pd.read_pickle(r'data_NA\NA_' + loc + 'test_data.pkl')
    test_data = pd.read_pickle(r'data_NA\NA_' + loc + 'val_data.pkl')

    return train_data, val_data, test_data

loc = ''
train_data, val_data, test_data = import_datasets(loc)

# use the training dataset to lear the OLS
# here I get the MB data
M_dot_silv = train_data['MB_Year']

# run the statsmodels and directly pick the variables out of the dataframe
X0 = train_data[['Annual_SF', 'TMPP']]

# don't forget to add the bias
X = sm.add_constant(X0)

# etimate the model parameters with statsmodel
sm_massbal_model = sm.OLS(M_dot_silv, X).fit()

print(sm_massbal_model.summary())

# now use the test dataset to predict 
X0_test = test_data[['Annual_SF', 'TMPP']]
X_test = sm.add_constant(X0_test)

# let the statsmodel predict the annual mass balance, based on the test data
MBA_test_predict = sm_massbal_model.predict(X_test)
MBA_test = test_data['MB_Year']

# calculate the test Mean Absolute Error like Tensorflow: loss = mean(abs(y_true - y_pred))

MAE = np.mean(np.abs(MBA_test - MBA_test_predict))
print('======== model performance =========')
print('MAE model: %.03f' % (MAE))

plt.plot(MBA_test, MBA_test_predict, 'o')
plt.plot([-2500, 1000], [-2500, 1000], '-k')
plt.title('Plot for NA' + loc)
plt.xlabel('data')
plt.ylabel('model')
plt.show()