import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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

# Απαραίτητος έλεγχος για 'null' τιμές στα δεδομένα μας.
print(df.isnull().sum())

def EDA(data):
	sns.pairplot(data, diag_kind='kde')
	#plt.savefig('pairplot.png')
	plt.show()
	plt.close()

	data.hist(figsize=(15,15))
	#plt.savefig('histogramm.png')
	plt.show()
	plt.close()

	sns.set(font_scale=1.15)
	plt.figure(figsize=(14, 10))
	sns.heatmap(data.corr(), vmax=.8, linewidths=0.01, square=True, annot=True, cmap="BuPu", linecolor="black")
	plt.title('Correlation between features')
	#plt.savefig('heatmap.png')
	plt.show()
	plt.close()
	
	return

EDA(df) #Συνάρτηση με τα πλοταρίσματα των χαρακτηριστικών για περισσότερη κατανόηση των features.
