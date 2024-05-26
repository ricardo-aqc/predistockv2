from django.core.management.base import BaseCommand
from django.core.cache import cache
import requests
import pandas as pd
from django.conf import settings  # Correct import for settings

class Command(BaseCommand):
    help = 'Precarga tickers de acciones y nombres de compañías en la caché'
    api_key = settings.API_KEY  # Gestiona de forma segura la clave API

    def getStockData(self, exchange, apiKey):
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

        # Filtra las columnas deseadas
        filteredData = [
            {key: row[key] for key in ['symbol', 'name', 'lastsale', 'netchange', 'pctchange', 'marketCap']}
            for row in dataRows
        ]
        return filteredData

    def handle(self, *args, **options):
        apiKey = 'f8tLyF1hdCqvSbH3JAzD'
        nasdaqData = self.getStockData('nasdaq', apiKey)
        nyseData = self.getStockData('nyse', apiKey)

        nasdaqDf = pd.DataFrame(nasdaqData)
        nyseDf = pd.DataFrame(nyseData)
        nasdaqDf['exchange'] = 'NASDAQ'
        nyseDf['exchange'] = 'NYSE'

        combinedDf = pd.concat([nasdaqDf, nyseDf], ignore_index=True)
        combinedDf['marketCap'] = pd.to_numeric(
            combinedDf['marketCap'].str.replace(',', '').str.replace('$', '').str.replace('B', 'e9').str.replace('M', 'e6'), 
            errors='coerce'
        )
        filteredDf = combinedDf[combinedDf['marketCap'] > 2e9]
        filteredDf.reset_index(drop=True, inplace=True)

        stocks = []
        for ticker, name in zip(filteredDf['symbol'].values, filteredDf['name'].values):
            stocks.append({"ticker": ticker, "name": name})

        # Guarda esta estructura de datos en la caché
        cache.set('stocks', stocks, timeout=None)
        self.stdout.write(self.style.SUCCESS('Tickers de acciones y nombres de compañías cargados exitosamente en la caché'))
