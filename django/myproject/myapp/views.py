from django.shortcuts import render
from django.core.cache import cache
from django.http import JsonResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from django.conf import settings
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from django.core.serializers.json import DjangoJSONEncoder
from datetime import datetime
from numpy import array
import json
import requests
from django.conf import settings

def homeView(request):
    return render(request, 'myapp/archivo.html')

def getStockData(exchange, apiKey):
    url = f'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&download=true&exchange={exchange}'
    headers = {
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'User-Agent': 'Java-http-client/',
        'x-api-key': apiKey
    }
    response = requests.get(url, headers=headers)
    jsonData = response.json()
    dataRows = jsonData['data']['rows']

    # Filter the desired columns
    filteredData = [{key: row[key] for key in ['symbol', 'name', 'lastsale', 'netchange', 'pctchange', 'marketCap']} for row in dataRows]
    return filteredData

def accionesView(request):
    apiKey = settings.API_KEY  # Securely manage the API key

    # Fetch data for NASDAQ and NYSE
    nasdaqData = getStockData('NASDAQ', apiKey)
    nyseData = getStockData('NYSE', apiKey)

    # Create DataFrames for NASDAQ and NYSE
    nasdaqDf = pd.DataFrame(nasdaqData)
    nyseDf = pd.DataFrame(nyseData)

    # Add an 'exchange' column to distinguish between NASDAQ and NYSE
    nasdaqDf['exchange'] = 'NASDAQ'
    nyseDf['exchange'] = 'NYSE'

    # Concatenate both DataFrames into one
    combinedDf = pd.concat([nasdaqDf, nyseDf], ignore_index=True)

    # Convert 'marketCap' to a numeric value after removing symbols and converting billions and millions
    combinedDf['marketCap'] = pd.to_numeric(
        combinedDf['marketCap'].str.replace(',', '').str.replace('$', '').str.replace('B', 'e9').str.replace('M', 'e6'),
        errors='coerce'
    )

    # Apply the market capitalization filter
    filteredDf = combinedDf[combinedDf['marketCap'] > 2e9]

    # Reset the index
    filteredDf.reset_index(drop=True, inplace=True)

    # Pass the processed data to the template
    context = {'stocks': filteredDf.to_dict('records')}
    return render(request, 'myapp/acciones.html', context)

def accionView(request, testMode=False):
    ticker = request.GET.get('ticker', '').upper()
    if not ticker:
        return JsonResponse({"error": "Ticker is required"}, status=400)

    if testMode:
        # Datos de prueba
        history = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=280),
            'Close': np.random.random(280) * 100,
            'High': np.random.random(280) * 110,
            'Low': np.random.random(280) * 90,
            'Volume': np.random.randint(1000, 1000000, size=280)
        }).set_index('Date')
    else:
        symbol = yf.Tickers(ticker)
        try:
            history = symbol.tickers[ticker].history(period='280d')
        except KeyError:
            return JsonResponse({"error": f"Ticker {ticker} not found"}, status=404)
    
    modelPath = 'myapp/trainedmodel/lstm9.h5'

    # Load LSTM model
    model = tf.keras.models.load_model(modelPath)    
    scaler = MinMaxScaler(feature_range=(0, 1))
    timeStep = 180

    def createDataset(data, timeStep=1):
        X, y = [], []
        for i in range(len(data) - timeStep):
            X.append(data[i:(i + timeStep)])
            y.append(data[i + timeStep])
        return np.array(X), np.array(y)

    history = history.drop(['Dividends', 'Stock Splits'], axis=1)
    closePrices = history['Close']
    X, y = createDataset(closePrices.values, timeStep)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    predictionLstmTest = scaler.inverse_transform(model.predict(X))
    dailyReturnsPred = np.diff(predictionLstmTest.flatten() + 0) / y[:-1]
    data = np.cumprod(1 + dailyReturnsPred)
    
    # Generate a date range for business days
    dateRange = closePrices[-100:-1].index

 # ARIMA -------------------------------------------------------
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.stattools import adfuller
    warnings.filterwarnings("ignore")

    def optimal_d(series):
        d=0
        p_value=adfuller(series)[1]
        while p_value > 0.05:
            d += 1
            p_value=adfuller(np.diff(series,n=d))[1]
        return d

    d_optimal = optimal_d(closePrices)
    total_values_bic=[]
    total_values_mse=[]

    def evaluar_modelo(dataset, p_values, d , q_values):
        dataset = dataset.astype("float32")
        best_score, best_cf = float("inf"), None
        for p in p_values:
            for q in q_values:
                order = (p,d,q)
            try:
                model = ARIMA(dataset,order=order)
                model_fit = model.fit()
                mse = mean_squared_error(dataset, model_fit.fittedvalues)
                bic = model_fit.bic
                total_values_bic.append(bic)
                total_values_mse.append(mse)
                if bic < best_score:
                    best_score, best_cf= bic, order
            except:
                continue
        return best_cf

    p_values = range(0,5)
    q_values = range(0,5)
    best_order = evaluar_modelo(closePrices,p_values,d_optimal,q_values)
    model=ARIMA(closePrices,order=(best_order[0], best_order[1], best_order[2]))
    model_fit=model.fit()
    print(model_fit.summary())
    arimaPredictions = model_fit.forecast(steps=15)

    # GARCH --------------------------------------------------------------
    from arch import arch_model
    returns = np.log(closePrices / closePrices.shift(1)).dropna()
    totalValuesBic = []
    totalValuesMse = []

    def evaluarModeloGarch(dataset, pValues, qValues):
        dataset = dataset.astype("float32")
        bestScore, bestCf = float("inf"), None
        for p in pValues:
            for q in qValues:
                order = (p, q)
                try:
                    model = arch_model(dataset, vol='GARCH', p=p, q=q, lags=1)
                    modelFit = model.fit(disp='off')
                    bic = modelFit.bic
                    totalValuesBic.append(bic)
                    if bic < bestScore:
                        bestScore, bestCf = bic, order
                except:
                    continue
        return bestCf

    # Call the function
    bestOrder = evaluarModeloGarch(returns, p_values, q_values)
    model = arch_model(returns, vol='GARCH', p=bestOrder[0], q=bestOrder[1], lags=1)
    results = model.fit(disp='off')
    forecasts = results.forecast(horizon=5)

    def obtenerRiesgo(forecast, bestOrder):
        tickerG = yf.Tickers("^GSPC")
        history = tickerG.tickers['^GSPC'].history(period='100d').drop(['Dividends', 'Stock Splits'], axis=1)
        returns = np.log(history['Close'] / history['Close'].shift(1)).dropna()
        model = arch_model(returns, vol='GARCH', p=bestOrder[0], q=bestOrder[1], lags=1)
        results = model.fit(disp='off')
        forecastHorizon = 5
        forecastsComparison = results.forecast(horizon=forecastHorizon)
        if forecast.mean.values[:,-1][0] > forecastsComparison.mean.values[:,-1][0]:
            # Alto riesgo
            return True
        else:
            return False
    
    # Informacion del dia
    from datetime import datetime
    import locale
    # Obtener la fecha actual
    locale.setlocale(locale.LC_TIME, 'es_ES.utf8')
    fechaActual = datetime.now()
    # Crear el string con el formato deseado
    dateForPrediction = fechaActual.strftime("%d de %B")
    lstmPrediction = predictionLstmTest[-1][0]
    lstmPrediction = round(lstmPrediction, 2)
    arimaPrediction = arimaPredictions.values[-1]
    arimaPrediction = round(arimaPrediction, 2)
    previousPrice = round(closePrices[-1], 2)
    diffLstm = abs(previousPrice - lstmPrediction)
    diffArima = abs(previousPrice - arimaPrediction)
    # Determinar la mejor predicción
    if diffLstm < diffArima:
        bestPrediction = lstmPrediction
    else:
        bestPrediction = arimaPrediction
    print(f"Previous Price: {previousPrice}")
    print(f"LSTM Prediction: {lstmPrediction}")
    print(f"ARIMA Prediction: {arimaPrediction}")
    # Calcular la desviación estándar de las predicciones y los precios reales
    stdPreciosReales = np.std(y[:-1])
    # Calcular los límites superior e inferior basados en dos desviaciones estándar de los precios reales
    upperLimit = previousPrice + 2 * stdPreciosReales
    lowerLimit = previousPrice - 2 * stdPreciosReales
    riesgo = ''
    if previousPrice > upperLimit or previousPrice < lowerLimit:
        riesgo = 'Alto Riesgo'
    else:
        if obtenerRiesgo(forecasts, bestOrder):
            riesgo = 'Alto Riesgo'
        else:
            riesgo = 'Bajo Riesgo'

    # Señal de compra
    from ta.volatility import BollingerBands

    indicator = BollingerBands(
        close=history['Close'],
        window=20,
        fillna=True)

    history['Bollinger_Hband'] = indicator.bollinger_hband()
    history['Bollinger_Lband'] = indicator.bollinger_lband()
    history['Bollinger_Mband'] = indicator.bollinger_mavg()

    from ta.momentum import StochasticOscillator

    indicator = StochasticOscillator(
        close=history['Close'],
        high=history['High'],
        low=history['Low'],
        window=14,
        fillna=True)

    history['SR'] = indicator.stoch()

    from ta.momentum import RSIIndicator

    indicator = RSIIndicator(
        close=history['Close'],
        window=14,
        fillna=True)

    history['RSI'] = indicator.rsi()

    from ta.trend import MACD

    indicator = MACD(
        close=history['Close'],
        window_slow=26,
        window_fast=12,
        window_sign=9,
        fillna=True)

    history['MACD'] = indicator.macd()

    from ta.trend import EMAIndicator

    indicator = EMAIndicator(
        close=history['Close'],
        window=14,
        fillna=True)

    history['EMA'] = indicator.ema_indicator()

    from ta.volume import OnBalanceVolumeIndicator

    indicator = OnBalanceVolumeIndicator(
        close=history['Close'],
        volume=history['Volume'],
        fillna=True)

    history['OBV'] = indicator.on_balance_volume()

    history15 = history[['Bollinger_Hband', 'Bollinger_Lband', 'Bollinger_Mband', 'SR', 'RSI', 'MACD', 'EMA', 'OBV', 'Close']].tail(15)

    def generarSenalesBollinger(bandasBollinger, preciosCierre):
        # Asegurar que la comparación es entre series y no incluye NaN de forma incorrecta
        senalCompra = ((preciosCierre < bandasBollinger['Bollinger_Lband']) &
                       (preciosCierre.shift(1) > bandasBollinger['Bollinger_Lband'].shift(1))).fillna(False)
        senalVenta = ((preciosCierre > bandasBollinger['Bollinger_Hband']) &
                      (preciosCierre.shift(1) < bandasBollinger['Bollinger_Hband'].shift(1))).fillna(False)
        return senalCompra.astype(bool), senalVenta.astype(bool)

    def generarSenalesRsi(rsi):
        senalCompra = ((rsi < 30) & (rsi.shift(1) > 30)).fillna(False)
        senalVenta = ((rsi > 70) & (rsi.shift(1) < 70)).fillna(False)
        return senalCompra.astype(bool), senalVenta.astype(bool)

    def generarSenalesMacd(macd):
        senalCompra = ((macd > 0) & (macd.shift(1) <= 0)).fillna(False)
        senalVenta = ((macd < 0) & (macd.shift(1) >= 0)).fillna(False)
        return senalCompra.astype(bool), senalVenta.astype(bool)

    def generarSenalesEma(ema, preciosCierre):
        senalCompra = (preciosCierre < ema).fillna(False)
        senalVenta = (preciosCierre > ema).fillna(False)
        return senalCompra.astype(bool), senalVenta.astype(bool)

    def generarSenalesObv(obv):
        senalCompra = (obv > obv.shift(1)).fillna(False)
        senalVenta = (obv < obv.shift(1)).fillna(False)
        return senalCompra.astype(bool), senalVenta.astype(bool)

    def evaluarCondiciones(senalCompra, senalVenta):
        # Evalúa las condiciones de compra y venta para determinar la acción final
        if senalCompra.sum() > senalVenta.sum():
            return 'Comprar'
        elif senalVenta.sum() > senalCompra.sum():
            return 'Vender'
        else:
            return 'Mantener'

    senalCompraBollinger, senalVentaBollinger = generarSenalesBollinger(history15[['Bollinger_Lband', 'Bollinger_Hband']], history15['Close'])
    senalCompraRsi, senalVentaRsi = generarSenalesRsi(history15['RSI'])
    senalCompraMacd, senalVentaMacd = generarSenalesMacd(history15['MACD'])
    senalCompraEma, senalVentaEma = generarSenalesEma(history15['EMA'], history15['Close'])
    senalCompraObv, senalVentaObv = generarSenalesObv(history15['OBV'])

    contadorCompra = senalCompraBollinger + senalCompraRsi + senalCompraMacd + senalCompraEma + senalCompraObv
    contadorVenta = senalVentaBollinger + senalVentaRsi + senalVentaMacd + senalVentaEma + senalVentaObv

    history15['Accion'] = 'Mantener'

    for idx in history15.index:
        accion = evaluarCondiciones(
            contadorCompra.loc[idx],
            contadorVenta.loc[idx]
        )
        history15.at[idx, 'Accion'] = accion

    closePrices15 = history15['Close'].values
    recommendedActions = history15['Accion'].values

    closePricesComprar = np.full(len(closePrices15), None, dtype=np.float64)
    closePricesVender = np.full(len(closePrices15), None, dtype=np.float64)
    
    for i, accion in enumerate(recommendedActions):
        if accion == 'Comprar':
            closePricesComprar[i] = closePrices15[i]
        elif accion == 'Vender':
            closePricesVender[i] = closePrices15[i]

    def nanToNone(array):
        return [None if isinstance(x, float) and np.isnan(x) else x for x in array]

    closePricesComprar = nanToNone(closePricesComprar)
    closePricesVender = nanToNone(closePricesVender)

    formattedDates = dateRange.strftime('%Y-%m-%d').to_list()
    last15Date = formattedDates[-15:]
    labels = formattedDates
        
    predictions = data.flatten().tolist()
    closePrices15 = closePrices15.flatten().tolist()
    comparison = predictionLstmTest.flatten().tolist()
    chartData = json.dumps({
        'labels': labels,
        'predictions': predictions,
    }, cls=DjangoJSONEncoder)
    
    actionsChartData = json.dumps({
        'labels': last15Date,
        'predictions': closePrices15,
        'pricesSale': closePricesVender,
        'pricesBuy': closePricesComprar
    }, cls=DjangoJSONEncoder)
    predChartData = json.dumps({
        'labels': labels,
        'predictions': comparison,
        'closePrices': y.flatten().tolist()
    }, cls=DjangoJSONEncoder)
    print(bestPrediction)
    context = {'ticker': ticker, 'chartData': chartData, 'riesgo': riesgo, 'actionsChartData': actionsChartData,
               'dateForPrediction': dateForPrediction, 'previousPrice': previousPrice,
               'bestPrediction': bestPrediction, 'recommendation': recommendedActions[-1],
               'predChartData': predChartData}
    return render(request, 'myapp/accion.html', context)

def searchView(request):
    query = request.GET.get('q', '').upper()
    stocks = cache.get('stocks', [])

    results = [stock for stock in stocks if query in stock['ticker'] or query in stock['name'].upper()][:10]

    return JsonResponse({"results": results})

# Create your views here.
