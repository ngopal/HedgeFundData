import requests
from bs4 import BeautifulSoup
import time, random, pickle, os
import datetime


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

def composeUrl(day, month, year):
  """ month --> 5 for may, not 05
      day   --> 2 for 2nd, 24 for 24th
      year  --> 2019    """
  return "http://www.cboe.com/data/current-market-statistics/daily-market-statistics-all-cboe?Dy="+str(day)+"&Mo="+str(month)+"&Yr="+str(year)

def getCBOEData(y, m, d):
  day_url = composeUrl(d, m, y)
  day = requests.get(day_url, headers = headers)
  soup = BeautifulSoup(day.text)
  tabledata = soup.findAll("article", {"class": "mktstat"})
  return(tabledata)

now = datetime.datetime.now()
for y in range(2016,2020):
  for m in range(1,13):
    for d in range(1,31):
      print(y, m, d)
      if y > int(now.year):
          print("year too far ahead")
          continue
      if y == int(now.year) and m > int(now.month):
          print("year fine, month too far ahead")
          continue
      if y == int(now.year) and m == int(now.month) and d > int(now.day):
          print("year fine, month fine, day too far ahead")
          continue
      filename = str(d)+"_"+str(m)+"_"+str(y)+"_cboe_futures.p"
      if filename in os.listdir("./cboe/"):
        print(filename, "exists, skipping")
      else:
        result = getCBOEData(y, m, d)
        pickle.dump([i.text.strip() for i in result], open( "./cboe/"+filename, "wb" ))
        print([i for i in result])
        waittime = random.randrange(1,5)
        print("WAIT TIME", waittime)
        time.sleep(waittime)
