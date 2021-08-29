import pandas
import numpy as np
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Calculate elapsed years
def year_until_today(born):
    if type(born) != float:
        born = datetime.strptime(born, "%Y-%m-%d").date()
        today = date.today()

        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    else:
        return 0

# Calculate elapsed days
def day_until_today(born):
    if type(born) != float:
        born = datetime.strptime(born, "%Y-%m-%d").date()
        today = date.today()
        delta = today - born

        return delta.days
    else:
        return 0

def main():
    # Read data from txt and assign to a dataframe
    consumidores = pandas.read_csv('1.consumidores.txt', delimiter='\t')

    # Calculate age
    consumidores['idade'] = consumidores['data_nascimento'].apply(year_until_today)
    consumidores['dias_uso'] = consumidores['primeiro_login'].apply(day_until_today)

    # Read data from txt and assign to a dataframe
    conversoes = pandas.read_csv('1.disparos_conversao.txt', delimiter='\t')

    # Merge dataframes
    df = pandas.merge(consumidores, conversoes, how='outer')

    # After merge the 'converteu' data type was changing to float, so i decided to force a conversion to int64
    df['converteu'] = df['converteu'].astype('Int64')

    # Get the values where 'converteu' is 1 or 0 in dataframe
    df_base = df.query('converteu == 0 or converteu == 1')
    df_test = df.query('converteu.isnull()')
    consumidor_id = df_test['consumidor_id'].values
    df_base = df_base[['renda', 'acessos', 'visualizacoes_divida', 'dias_uso', 'idade', 'converteu']]
    df_test = df_test[['renda', 'acessos', 'visualizacoes_divida', 'dias_uso', 'idade']]
    X = df_base.iloc[:,:-1].values
    y = df_base.iloc[:, 5].values

    # Convert 'renda' to number
    labelEncoder_renda = LabelEncoder()
    X[:,0] = labelEncoder_renda.fit_transform(X[:,0])

    # Convert data type
    X = np.vstack(X[:, :]).astype(np.float)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Doing logistic regression classifier
    classifier = RandomForestClassifier()
    y_train = y_train.astype('int')
    classifier.fit(X_train, y_train)

    # Predict
    y_pred = classifier.predict(X_test)

    # Change data type to make operation between them
    y_test = y_test.to_numpy()
    y_test = y_test.astype(np.int32)

    # Prints with some results about model
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix: \n', cm)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('\nAccuracy score: ', accuracy)
    precision = metrics.precision_score(y_test, y_pred)
    print("\nPrecision score:",precision)
    recall = metrics.recall_score(y_test, y_pred) 
    print("\nRecall score:",recall)

    # Convert 'renda' to number
    labelEncoder_renda = LabelEncoder()
    df_test['renda'] = labelEncoder_renda.fit_transform(df_test['renda'])
    
    # Predict using other data, without 'estrategia' and 'converteu'
    result = classifier.predict(df_test)

    # Save all results
    list = []
    for i in range(len(consumidor_id)):
        list.append([consumidor_id[i], result[i]])
    
    pandas.DataFrame(list).to_csv('resultado.txt', header=['consumidor_id', 'previsao'], index=None, sep='\t', mode='a')


if __name__ == '__main__':
    main()