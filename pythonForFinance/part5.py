import bs4 as bs
import pickle
import requests

def saveSP500():
    req = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(req.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open('resources/sp500tickers.pickle', 'wb') as file:
        pickle.dump(tickers, file)
        
    return tickers






