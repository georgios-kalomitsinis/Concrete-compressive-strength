# Εισαγωγή απαραίτητων βιβλιοθηκών για την οπτικοποιήση και εισαγωγή των δεδομένων.
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Κάνω import το csv file, για να διαβάσω τα δεδομένα της άσκησης.
df = pd.read_csv('Concrete_Data.csv', sep=",")

# Κάνω import το excel_file, με στόχο να διαβάσω τα ονόματα των δεδομένων.
xls_file = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')
names_of_cols = list(xls_file.columns)

# Διαδικασία για να διαβάσω τα ονόματα των στήλων, χωρίς τις μονάδες μέτρησεις.
names = []
for i in names_of_cols:
	names.append(i.split('(')[0])

# Ονομάζω τις στήλες του csv file, με τα ονόματα που έχω εξάγει.
dir_names = {}
for i, name in enumerate(list(df.columns)):
	dir_names[name] = names[i]

df = df.rename(columns = dir_names)
print(df.head(5))

#Απαραίτητος έλεγχος για 'null' τιμές στα δεδομένα μας.
print(df.isnull().sum())

# Όπως αναφέρεται και από την εκφώνηση της άσκησης η μεταβλητή που προσπαθούμε να μοντελοποιήσουμε αντιστοιχεί
# στην τελευταία στήλη, όποτε το Χ αντιστοιχεί στις στήλες εκτός της τελευταίας ενώ το y στη τελευταία.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(X.shape, y.shape)

# Θα χρησιμοποιήσουμε το 70% του συνόλου δεδομένων για εκπαίδευση (με τη σειρά που δίνεται)
# και το υπόλοιπο 30% για αξιολόγηση (testing), όπως αναφέρεται και από την εκφώνηση.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Κάνουμε scaling τα features
sc_features = StandardScaler()
X_train = sc_features.fit_transform(X_train)
X_test = sc_features.transform(X_test)

#Συναρτήσεις για την ευρεση των train_set error kai test_set error
def train_error(X_train, y_train, model):
	predictions = model.predict(X_train)
	mse = mean_squared_error(y_train, predictions)
	mae = mean_absolute_error(y_train, predictions)
	mape = np.mean(np.abs(((y_train - predictions) / y_train))) * 100
	return mse, mae, mape

def test_error(X_test, y_test, model):
	predictions = model.predict(X_test)
	mse = mean_squared_error(y_test, predictions)
	mae = mean_absolute_error(y_test, predictions)
	mape = np.mean(np.abs(((y_test - predictions) / y_test))) * 100
	return mse, mae, mape

def metrics(X_train, y_train, X_test, y_test, model):
	model.fit(X_train, y_train)
	train_error_mse, train_error_mae, train_error_mape = train_error(X_train, y_train, model)
	test_error_mse, test_error_mae, test_error_mape = test_error(X_test, y_test, model)
	return  train_error_mse, train_error_mae, train_error_mape, test_error_mse, test_error_mae, test_error_mape

#Ορισμός μοντέλων
lr_model_1 = LinearRegression()

train_error_mse_lr, train_error_mae_lr, train_error_mape_lr, test_error_mse_lr, test_error_mae_lr, test_error_mape_lr = metrics(X_train, y_train,
                                                                                                                               X_test,  y_test,
                                                                                                                               lr_model_1)


print('##################     Α ΕΡΩΤΗΜΑ      ########################')
#Προβολή των αποτελεσμάτων για το γραμμικο μοντέλο
print('-' * 80)
print('\t\t\t\t\t\t\t\t\t\t MSE \t MAE \t MAPE')
print('Train_error Linear Regression \t {:.2f}  {:.2f} \t {:.2f}'.format(round(np.mean(train_error_mse_lr), 3),
                                                                                round(np.mean(train_error_mae_lr), 3),
                                                                                round(np.mean(train_error_mape_lr), 3)))
print('Test_error Linear Regression  \t {:.2f}   {:.2f} \t {:.2f}'.format(round(np.mean(test_error_mse_lr), 3),
                                                                                round(np.mean(test_error_mae_lr), 3),
                                                                               round(np.mean(test_error_mape_lr), 3)))

print('-' * 80)

alphas = [1e-3, 1e-2, 1e-1, 1, 5, 10, 100, 1000]
for alpha in alphas:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=False)
    lasso_model = Lasso(alpha = alpha)
    ridge_model = Ridge(alpha = alpha)

    train_error_mse_lasso, train_error_mae_lasso, train_error_mape_lasso, test_error_mse_lasso, test_error_mae_lasso, test_error_mape_lasso = metrics(X_train, y_train, X_test, y_test, lasso_model)
    train_error_mse_ridge, train_error_mae_ridge, train_error_mape_ridge, test_error_mse_ridge, test_error_mae_ridge, test_error_mape_ridge = metrics(X_train, y_train, X_test, y_test, ridge_model)
    print('-' * 80)
    print('Για alpha = {}:'.format(alpha))
    print("\t\t\t\t\t\t\t\t MSE \t MAE \t MAPE")
    print('Train_error Mοντέλο Lasso \t\t {:.2f}  {:.2f} \t {:.2f}'.format(train_error_mse_lasso, train_error_mae_lasso, train_error_mape_lasso))
    print('Test_error Mοντέλο Lasso \t\t {:.2f}   {:.2f} \t {:.2f}'.format(test_error_mse_lasso, test_error_mae_lasso, test_error_mape_lasso))
    print('-   -   ' * 10)
    print('Train_error Mοντέλο Ridge \t\t {:.2f}  {:.2f} \t {:.2f}'.format(train_error_mse_ridge, train_error_mae_ridge, train_error_mape_ridge))
    print('Test_error Mοντέλο Ridge \t\t {:.2f}   {:.2f} \t {:.2f}'.format(test_error_mse_ridge, test_error_mae_ridge, test_error_mape_ridge))
    print('-' * 80)

print('################             Γ   ΕΡΩΤΗΜΑ              ######################')
cv = KFold(n_splits=10, shuffle=True, random_state=42)

train_mse_lr = []
test_mse_lr = []

train_mae_lr = []
test_mae_lr = []

train_mape_lr = []
test_mape_lr = []
for train_id, test_id in cv.split(X, y):
	X_train, X_test = X.iloc[train_id], X.iloc[test_id]
	y_train, y_test = y.iloc[train_id], y.iloc[test_id]

	sc_features = StandardScaler()
	X_train = sc_features.fit_transform(X_train)
	X_test = sc_features.transform(X_test)

	lr_model_2 = LinearRegression()
	train_mse_lr_, train_mae_lr_, train_mape_lr_, test_mse_lr_, test_mae_lr_, test_mape_lr_ = metrics(X_train, y_train, X_test, y_test, lr_model_2)

	train_mse_lr.append(train_mse_lr_)
	test_mse_lr.append(test_mse_lr_)

	train_mae_lr.append(train_mae_lr_)
	test_mae_lr.append(test_mae_lr_)

	train_mape_lr.append(train_mape_lr_)
	test_mape_lr.append(test_mape_lr_)

print("Train Errors \t MSE \t\t\t\t\t MAE \t\t\t\t\t MAPE")
print("\t\t\t\t Avg \t\t Std \t\t Avg \t Std \t\t Avg \t\t Std")
print('Linear: \t\t {} \t {} \t\t {}\t {} \t\t {} \t {}'.format(round(np.mean(train_mse_lr), 3), round(np.std(train_mse_lr), 3),
														round(np.mean(train_mae_lr), 3), round(np.std(train_mae_lr), 3),
														round(np.mean(train_mape_lr), 3), round(np.std(train_mape_lr), 3)))
print('\n')
print("Test Errors")
print('Linear: \t\t {} \t {} \t {}\t {} \t\t {} \t {}'.format(round(np.mean(test_mse_lr), 3), round(np.std(test_mse_lr), 3),
														round(np.mean(test_mae_lr), 3), round(np.std(test_mae_lr), 3),
														round(np.mean(test_mape_lr), 3), round(np.std(test_mape_lr), 3)))

print('K-Fold Results')
print('\n')
for alpha in alphas:
	train_mse_lasso = []
	test_mse_lasso = []

	train_mae_lasso = []
	test_mae_lasso = []

	train_mape_lasso = []
	test_mape_lasso = []

	train_mse_ridge = []
	test_mse_ridge = []

	train_mae_ridge = []
	test_mae_ridge = []

	train_mape_ridge = []
	test_mape_ridge = []

	for train_id, test_id in cv.split(X, y):
		X_train, X_test = X.iloc[train_id], X.iloc[test_id]
		y_train, y_test = y.iloc[train_id], y.iloc[test_id]

		sc_features = StandardScaler()
		X_train = sc_features.fit_transform(X_train)
		X_test = sc_features.transform(X_test)


		lasso = Lasso(alpha=alpha)
		ridge = Ridge(alpha=alpha)

		train_mse_lasso_, train_mae_lasso_, train_mape_lasso_, test_mse_lasso_, test_mae_lasso_, test_mape_lasso_ = metrics(X_train, y_train,
																															 X_test, y_test,
																															 lasso)
		train_mse_ridge_, train_mae_ridge_, train_mape_ridge_, test_mse_ridge_, test_mae_ridge_, test_mape_ridge_ = metrics(X_train, y_train,
																															 X_test, y_test,
																															 ridge)

		train_mse_lasso.append(train_mse_lasso_)
		test_mse_lasso.append(test_mse_lasso_)

		train_mae_lasso.append(train_mae_lasso_)
		test_mae_lasso.append(test_mae_lasso_)

		train_mape_lasso.append(train_mape_lasso_)
		test_mape_lasso.append(test_mape_lasso_)

		train_mse_ridge.append(train_mse_ridge_)
		test_mse_ridge.append(test_mse_ridge_)

		train_mae_ridge.append(train_mae_ridge_)
		test_mae_ridge.append(test_mae_ridge_)

		train_mape_ridge.append(train_mape_ridge_)
		test_mape_ridge.append(test_mape_ridge_)

	train_mse_lasso = np.array(train_mse_lasso)
	test_mse_lasso = np.array(test_mse_lasso)

	train_mae_lasso = np.array(train_mae_lasso)
	test_mae_lasso = np.array(test_mae_lasso)

	train_mape_lasso = np.array(train_mape_lasso)
	test_mape_lasso = np.array(test_mape_lasso)

	train_mse_ridge = np.array(train_mse_ridge)
	test_mse_ridge = np.array(test_mse_ridge)

	train_mae_ridge = np.array(train_mae_ridge)
	test_mae_ridge = np.array(test_mae_ridge)

	train_mape_ridge = np.array(train_mape_ridge)
	test_mape_ridge = np.array(test_mape_ridge)

	print('-' * 80)
	print('For alpha = {}:'.format(alpha))
	print("Train Errors \t MSE \t\t\t\t\t MAE \t\t\t\t\t MAPE")
	print("\t\t\t Avg \t\t Std \t\t Avg \t Std \t\t Avg \t\t Std")
	print('Lasso: \t\t {} \t {} \t\t {}\t {} \t\t {} \t {}'.format(round(np.mean(train_mse_lasso), 3), round(np.std(train_mse_lasso), 3),
														round(np.mean(train_mae_lasso), 3), round(np.std(train_mae_lasso), 3),
														round(np.mean(train_mape_lasso), 3), round(np.std(train_mape_lasso), 3)))

	print('Ridge: \t\t {} \t {} \t\t {}\t {} \t\t {} \t {}'.format(round(np.mean(train_mse_ridge), 3), round(np.std(train_mse_ridge), 3),
														round(np.mean(train_mae_ridge), 3), round(np.std(train_mae_ridge), 3),
														round(np.mean(train_mape_ridge), 3), round(np.std(train_mape_ridge), 3)))

	print('\n')
	print('Test Errors')
	print('Lasso: \t\t {} \t {} \t {}\t {} \t\t {} \t {}'.format(round(np.mean(test_mse_lasso), 3), round(np.std(test_mse_lasso), 3),
														round(np.mean(test_mae_lasso), 3), round(np.std(test_mae_lasso), 3),
														round(np.mean(test_mape_lasso), 3), round(np.std(test_mape_lasso), 3)))

	print('Ridge: \t\t {} \t {} \t {}\t {} \t\t {} \t {}'.format(round(np.mean(test_mse_ridge), 3), round(np.std(test_mse_ridge), 3),
														round(np.mean(test_mae_ridge), 3), round(np.std(test_mae_ridge), 3),
														round(np.mean(test_mape_ridge), 3), round(np.std(test_mape_ridge), 3)))

print('-' * 80)
print('#############         Δ   ΕΡΩΤΗΜΑ          ######################')
print('Polyonomial regression')
print('\n')

def test_poly_regression(X_train, y_train, X_test, y_test, n=2):
	features = []
	count = 0
	for i in range(1,n+1):
		poly = PolynomialFeatures(degree = i)
		X_poly_train = poly.fit_transform(X_train)
		X_poly_test = poly.transform(X_test)
		lr = LinearRegression()
		results = lr.fit(X_poly_train, y_train)

		train_mse_pol, train_mae_pol, train_mape_pol, test_mse_pol, test_mae_pol, test_mape_pol = metrics(X_poly_train, y_train, X_poly_test,  y_test, lr)

		print('-'*90)
		print('-'*90)
		print('For '+str(i)+'polyonomial degree:')
		print('Training Info:')
		print('MSE: '+str(round(train_mse_pol, 3))+', MAE: '+str(round(train_mae_pol, 3))+', MAPE: '+str(round(train_mape_pol, 3))+', R2-Score: '+str(round(r2_score(y_train, lr.predict(X_poly_train)), 3)))
		print('-   -'*18)
		print('Testing Info:')
		print('MSE: '+str(round(test_mse_pol, 3))+', MAE: '+str(round(test_mae_pol, 3))+', MAPE: '+str(round(test_mape_pol, 3))+', R2-Score: '+str(round(r2_score(y_test, lr.predict(X_poly_test)), 3)))
		
		

		X_features = np.vstack((X_poly_train, X_poly_test)) # Για κάθετη στοίχηση των features για να γίνεται σωστή αντιστοίχιση

		features.append(X_features)

	print(len(features), features) #Tπώνω τον μέγεθος του ζητούμενου πίνακα, καθώς και τον ίδιο 
	

	# Ο πίνακας με τα ζητούμενα χαρακτηριστικά είναι στο αρχείο με όνομα features.txt
	with open('features.txt', 'w+') as file:
		file.write(str(features))

test_poly_regression(X_train, y_train, X_test, y_test, n = 10)
