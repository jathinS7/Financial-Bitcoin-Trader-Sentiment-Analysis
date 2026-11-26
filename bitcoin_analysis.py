import pandas as pd
import numpy as np

# Expects the two CSVs in the same folder as this script (see README)
senti = pd.read_csv('fear_greed_index.csv')
trades = pd.read_csv('historical_data.csv')

print(senti.head())
print(trades.head())

print(senti.shape)
print(trades.shape)

print(trades['Coin'].value_counts())

print(trades['Timestamp IST'].dtype)
print(trades['Timestamp IST'].head(10))

trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], format = '%d-%m-%Y %H:%M')

trades['date'] = trades['Timestamp IST'].dt.date

senti['date'] = pd.to_datetime(senti['date']).dt.date

print(trades[['Timestamp IST', 'date']].head())
print(trades['date'].min() , 'to' , trades['date'].max())


merged = trades.merge(senti[['date', 'classification', 'value']], on='date', how='left')

print(merged['classification'].isna().sum())

print(merged[['date', 'Coin' , 'Side', 'Closed PnL', 'value', 'classification']].head(9))

closed = merged[merged['Closed PnL'] != 0].copy()

real_directions = ['Open Long', 'Close Long' , 'Open Short' , 'Close Short','Buy', 'Sell']
closed = closed[closed['Direction'].isin(real_directions)]

print(closed.shape)
print(closed['Direction'].value_counts()
      )

closed['is_win'] = closed['Closed PnL'] > 0 

print(closed[['Closed PnL' , 'is_win']].head(9))

win_rate = closed.groupby('classification')['is_win'].mean()
print(win_rate)

closed = closed[closed['Size USD'] > 0].copy()
closed['pnl_pct'] = closed['Closed PnL'] / closed['Size USD']

print(closed['pnl_pct'].describe())

print(closed.groupby('classification')['pnl_pct'].agg(['mean' ,'std', 'median']))





print(closed.nsmallest(5, 'pnl_pct')[['Coin', 'Size USD', 'Closed PnL', 'classification' , 'pnl_pct']])
print(closed.nlargest(5, 'pnl_pct')[['Coin', 'Size USD', 'Closed PnL', 'pnl_pct' , 'classification']])

lower = closed['pnl_pct'].quantile(0.01)
upper = closed['pnl_pct'].quantile(0.99)

print('cutoffs:' , lower, upper)

closed['pnl_pct_capped'] = closed['pnl_pct'].clip(lower, upper)

print(closed['pnl_pct_capped'].describe())
print(closed.groupby('classification')['pnl_pct_capped'].agg(['mean' , 'median', 'std']))



acnt = closed.groupby(['classification', 'Account'])['Closed PnL'].sum().reset_index()

def top_acnt(group):
    total = group['Closed PnL'].sum()
    top = group['Closed PnL'].max()
    return pd.Series({'total_pnl' : total , 'top_acnt' : top , 'top_share' : (top/total*100) if total != 0 else 0 })

result = acnt.groupby('classification').apply(top_acnt)
print(result)



closed['log_size_USD'] = np.log1p(closed['Size USD'])

closed['is_taker'] = closed['Crossed'].astype(int)

closed['fee_pct'] = closed['Fee'] / closed['Size USD']

print(closed[['is_win', 'value', 'fee_pct', 'log_size_USD', 'is_taker']].describe())

closed = closed.dropna(subset=['value', 'log_size_USD', 'fee_pct'])
print(closed.shape)

dummies = pd.get_dummies(closed['Direction'], prefix='dir')
print(dummies.head())

top_coins = closed['Coin'].value_counts().head(10).index

closed['coin_grouped'] = closed['Coin'].where(closed['Coin'].isin(top_coins) , 'others')
print(closed['coin_grouped'].value_counts().head(10))

coin_dummies = pd.get_dummies(closed['coin_grouped'], prefix='coin')


features = pd.concat([
    closed[['is_taker', 'fee_pct', 'log_size_USD', 'value']].reset_index(drop=True),
    dummies.reset_index(drop = True),
    coin_dummies.reset_index(drop=True)




], axis=1)


target = closed['is_win'].reset_index(drop = True)

print(features.shape, target.shape)
print(features.columns.tolist())
print(features.isna().sum())


from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(features, target, test_size = 0.2, random_state = 42, stratify = target)


print('train', X_train.shape , 'test',X_test.shape)
print(y_train.mean())
print(y_test.mean())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix, classification_report


rf = RandomForestClassifier( n_estimators = 250, max_depth = 8, random_state = 42, class_weight= 'balanced' )
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

print('accuracy' , accuracy_score(y_test, y_pred))
print('precision', precision_score(y_test, y_pred))
print('recall', recall_score(y_test,y_pred))

print('f1score' , f1_score(y_test, y_pred))
print('roc_auc' , roc_auc_score(y_test, y_proba))
print('confusion_matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('train accuracy', rf.score(X_train, y_train))
      



imp = pd.Series(rf.feature_importances_,index= features.columns).sort_values(ascending = False)
print(imp)




greed_wins = closed[closed['classification']=='Extreme Greed']['is_win']

other = closed[closed['classification']!='Extreme Greed']['is_win']

from statsmodels.stats.proportion import proportions_ztest
counts = [greed_wins.sum(), other.sum()]
nobs =[ len(greed_wins), len(other) ]
stat , pval = proportions_ztest(counts , nobs)
print(f'z-stats= {stat:.3f}, p-value={pval:.6f}')



      

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

accnts = closed.groupby('Account').agg(
    avg_size =('log_size_USD','mean'),
    win_rate = ('is_win','mean'),
    avg_fee_pct = ('fee_pct', 'mean'),
    taker_ratio = ('is_taker' , 'mean'),
    n_trades = ('is_win', 'count')


).reset_index()

scaler = StandardScaler()
X_cluster = scaler.fit_transform(accnts[['avg_size', 'win_rate', 'avg_fee_pct', 'taker_ratio']])

kmeans = KMeans(n_clusters= 3, random_state =42, n_init = 10)
accnts['cluster'] = kmeans.fit_predict(X_cluster)

print(accnts.groupby('cluster')[['avg_size', 'win_rate', 'avg_fee_pct', 'taker_ratio']].mean())
                                 

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, features, target, cv = 5, scoring = 'roc_auc')
print(cv_scores, cv_scores.mean())


from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle= True, random_state= 42)
cv_scores = cross_val_score(rf, features, target, cv= cv, scoring='roc_auc')

print(cv_scores , cv_scores.mean())

print('train AUC:', roc_auc_score(y_train, rf.predict_proba(X_train)[:,1]))
