import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score


#otvaranje podataka
df_p2 = pd.read_csv("oop2_rezultati_jun.csv", sep=';')
df_m2 = pd.read_csv("EE Matematika 2 jun 2021.csv", sep=';')
df_lab = pd.read_csv("13E071LOE_Jun_2021_Rezultati-converted.csv", sep=';')

#sredjivanje p2: gledamo sam indekse vece od 2020, potrebne kolone su samo Indeks, Prezime, Ukupno poena
df_p2 = df_p2[df_p2.Indeks > '2020/0000']
df_p2 = df_p2.reset_index(drop='True')
df_p2 = df_p2[['Indeks', "Prezime i ime", 'Ukupno']]
indeksi = df_p2['Indeks'].values
for i in range(len(indeksi)):
    df_p2.loc[i,'Indeks']=int(indeksi[i].split('/')[1])
df_p2=df_p2.rename(columns={'Indeks':'Индекс',"Prezime i ime": "Презиме и име"})

#sredjivanje m2
df_m2 = df_m2[['Индекс', 'Презиме и име', 'Укупно']]
df_m2 = df_m2[df_m2.Индекс > '2020/0000']
df_m2 = df_m2.reset_index(drop='True')
indeksi = df_m2['Индекс'].values
for i in range(len(indeksi)):
    df_m2.loc[i,'Индекс'] = int(indeksi[i].split('/')[1])

#sredjivanje lab
df_lab = df_lab[['Индекс', 'Име', 'Оцена']]
df_lab = df_lab[df_lab.Индекс > '2020/0000']
df_lab = df_lab.reset_index(drop='True')
indeksi = df_lab['Индекс'].values
for i in range(len(indeksi)):
    df_lab.loc[i,'Индекс'] = int(indeksi[i].split('/')[1])
df_lab = df_lab.rename(columns={'Име':'Презиме и име'})

#merge-ovanje data frame-ova za m2, p2, lab
df_m2_lab = pd.merge(df_lab,df_m2,how ='inner',on=['Индекс','Презиме и име'])
df_sve=pd.merge(df_m2_lab,df_p2,how='inner',on=['Индекс'])

#sredjivanje zajednickog data frame-a
df_sve = df_sve[['Индекс','Презиме и име_x','Укупно','Ukupno','Оцена']]
df_sve=df_sve[df_sve.Оцена>4]
df_sve = df_sve.reset_index(drop=True)

ukupno = df_sve['Ukupno'].values
for i in range(len(ukupno)):
    df_sve.loc[i,'Ukupno'] = float(ukupno[i].replace(',','.'))

#plot-ovanje zavisnosti Indeksa od ostalih kolona
plt.figure(figsize=(8,5))
x_data, y_data = (df_sve["Индекс"].values, df_sve["Оцена"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('Оцена')
plt.xlabel('Индекс')
plt.show()

msk = np.random.rand(len(df_sve)) < 0.8
train = df_sve[msk]
test = df_sve[~msk]

regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['Индекс','Укупно','Ukupno']])
y_train = np.asanyarray(train[['Оцена']])
regr.fit(x_train,y_train)
plt.figure(figsize=(8,5))
x_data, y_data = (df_sve["Индекс"].values, df_sve["Оцена"].values)
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data,x_data*regr.coef_[0][1]+regr.intercept_[0])
plt.ylabel('Оцена')
plt.xlabel('Индекс')
plt.show()

y_hat= regr.predict(test[['Индекс','Укупно','Ukupno']])
x = np.asanyarray(test[['Индекс','Укупно','Ukupno']])
y = np.asanyarray(test[['Оцена']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

print('Variance score: %.2f' % regr.score(x, y))
print("R2 score: %.2f" %r2_score(np.round(y_hat,0),y))

df_sve.to_csv("objedinjeno.csv",index=False,encoding='UTF-8')
coef = [-0.00222292, 0.02062339, 0.02574563]
intercept = [5.90809568]


