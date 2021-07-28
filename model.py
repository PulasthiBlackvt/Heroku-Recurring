import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle


from sklearn.preprocessing import LabelEncoder

# read csv
df = pd.read_csv('Recurring Transactions.csv', dtype='str', error_bad_lines=False, low_memory=False)
df.head()

df['isRecurring'] = df['isRecurring'].fillna(0)

df = df.fillna("0")

df = df.drop(columns=['account_id', 'userId', 'category_id', 'pending_transaction_id', 'transaction_id'])

df['isRecurring'] = df['isRecurring'].astype("int")

df['day'] = df['day'].astype("int")

df['amount']=df['amount'].astype(float)

dict = {}
count = 0
for x in df['category']:
    if x not in dict:
        dict[x] = count
        count = count + 1

df['category'] = df['category'].apply(lambda x: dict.get(x))
#df_Cat = pd.DataFrame(list(dict.items()),columns = ['key','value'])
#df_Cat.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\Cat_Json.json')

dict_Qu = {}
count = 0
for x in df['quirkCategory']:
    if x not in dict_Qu:
        dict_Qu[x] = count
        count = count + 1

df['quirkCategory'] = df['quirkCategory'].apply(lambda x: dict_Qu.get(x))
#df_Qu = pd.DataFrame(list(dict_Qu.items()),columns = ['key','value'])
#df_Qu.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\Qu_Json.json')

dict_Name = {}
count = 0
for x in df['name']:
    if x not in dict_Name:
        dict_Name[x] = count
        count = count + 1

df['name'] = df['name'].apply(lambda x: dict_Name.get(x))
#df_Name = pd.DataFrame(list(dict_Name.items()),columns = ['key','value'])
#df_Name.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\Name_Json.json')

dict_iso = {}
count = 0
for x in df['iso_currency_code']:
    if x not in dict_iso:
        dict_iso[x] = count
        count = count + 1

df['iso_currency_code'] = df['iso_currency_code'].apply(lambda x: dict_iso.get(x))
#df_iso = pd.DataFrame(list(dict_iso.items()),columns = ['key','value'])
#df_iso.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\iso_Json.json')

dict_MN = {}
count = 0
for x in df['merchant_name']:
    if x not in dict_MN:
        dict_MN[x] = count
        count = count + 1

df['merchant_name'] = df['merchant_name'].apply(lambda x: dict_MN.get(x))
#df_MN = pd.DataFrame(list(dict_MN.items()),columns = ['key','value'])
#df_MN.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\MN_Json.json')

dict_PC = {}
count = 0
for x in df['payment_channel']:
    if x not in dict_PC:
        dict_PC[x] = count
        count = count + 1

df['payment_channel'] = df['payment_channel'].apply(lambda x: dict_PC.get(x))
#df_PC = pd.DataFrame(list(dict_PC.items()),columns = ['key','value'])
#df_PC.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\PC_Json.json')

dict_PM = {}
count = 0
for x in df['payment_meta']:
    if x not in dict_PM:
        dict_PM[x] = count
        count = count + 1

df['payment_meta'] = df['payment_meta'].apply(lambda x: dict_PM.get(x))
#df_PM = pd.DataFrame(list(dict_PM.items()),columns = ['key','value'])
#df_PM.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\PM_Json.json')

dict_TC = {}
count = 0
for x in df['transaction_code']:
    if x not in dict_TC:
        dict_TC[x] = count
        count = count + 1

df['transaction_code'] = df['transaction_code'].apply(lambda x: dict_TC.get(x))
#df_TC = pd.DataFrame(list(dict_TC.items()),columns = ['key','value'])
#df_TC.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\TC_Json.json')

dict_TT = {}
count = 0
for x in df['transaction_type']:
    if x not in dict_TT:
        dict_TT[x] = count
        count = count + 1

df['transaction_type'] = df['transaction_type'].apply(lambda x: dict_TT.get(x))
#df_TT = pd.DataFrame(list(dict_TT.items()),columns = ['key','value'])
#df_TT.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\TT_Json.json')

dict_Y = {}
count = 0
for x in df['year']:
    if x not in dict_Y:
        dict_Y[x] = count
        count = count + 1

df['year'] = df['year'].apply(lambda x: dict_Y.get(x))
#df_Y = pd.DataFrame(list(dict_Y.items()),columns = ['key','value'])
#df_Y.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\Y_Json.json')

dict_M = {}
count = 0
for x in df['month']:
    if x not in dict_M:
        dict_M[x] = count
        count = count + 1

df['month'] = df['month'].apply(lambda x: dict_M.get(x))
#df_M = pd.DataFrame(list(dict_M.items()),columns = ['key','value'])
#df_M.to_json(r'I:\Blackvt 1\CSV files\Dictionaries\M_Json.json')

df2 = df.filter(['quirkCategory','category','name','day','isRecurring','iso_currency_code','merchant_name','payment_channel','payment_meta','transaction_code','transaction_type','year','month','amount'],axis=1)


from sklearn.model_selection import train_test_split

train, test = train_test_split(df2, test_size=0.15)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    train.drop('isRecurring', axis=1),
    train['isRecurring'],
    test_size=0.1)

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(learning_rate=0.3, n_estimators=80,max_depth=13, min_samples_split=1800,min_samples_leaf=40, subsample=0.8, random_state=10, max_features=3).fit(train.drop('isRecurring', axis=1), train['isRecurring'])

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, y_test)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, test.drop('isRecurring', axis=1), test['isRecurring'])

