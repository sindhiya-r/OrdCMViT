import pandas as pd
csv='./BCMID/BCMID_labels.csv'
df=pd.read_csv(csv,header=None)
df.columns=['patient_id','birads','binary_label']
print('total rows',len(df))
print('null birads',df['birads'].isna().sum())
print('unique sample',df['birads'].unique()[:20])
print(df.head())