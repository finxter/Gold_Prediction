import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('gold.csv')
# select the Close data
df = data[['Close']]

# add extra features
df['weekly_mean'] = data.Close.rolling(window=7).mean()
df['monthly_mean'] = data.Close.rolling(window=30).mean()
df['quarterly_mean'] = data.Close.rolling(window=90).mean()
df['yearly_mean'] = data.Close.rolling(window=365).mean()

# add the target variable
df['next_day_price'] = data.Close.shift(-1)
df = df.dropna()

# define independent variable
X = df[['weekly_mean', 'monthly_mean', 'quarterly_mean', 'yearly_mean']]

# define dependent variable
target = df['next_day_price']

# normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(X)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=0)

# define xgboost model
model = XGBRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
