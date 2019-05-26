import cbpro
import dateutil.parser
from datetime import datetime, timedelta
import time
import os, sys
import pickle

public_client = cbpro.PublicClient()

# Pull data
# https://docs.pro.coinbase.com/#rate-limits
end = datetime.utcnow().isoformat() #Need to confirm time of CB Source
start = (dateutil.parser.parse(end) - timedelta(hours=0, minutes=60)).isoformat()
data = []
print(start, end)
d = public_client.get_product_historic_rates('ETH-USD', granularity=60, start=start, end=end)
result = [(datetime.utcfromtimestamp(int(e[0])).strftime('%Y-%m-%d %H:%M:%S'),e) for e in d]
print(result)
pfilename = "-".join(end.replace(":","-").split(".")[0].split("-")[:-2]) + ".p"
pickle.dump( result, open("./data/files/eth/"+pfilename, "wb") )

# Find lowest unix epoch time. Save as new end variable
while True:
  pfilename = "-".join(end.replace(":","-").split(".")[0].split("-")[:-2]) + ".p"
  if pfilename not in os.listdir("./data/files/eth/"):
    print(pfilename, start, end)
    d = public_client.get_product_historic_rates('ETH-USD', granularity=60, start=start, end=end)
    result = [(datetime.utcfromtimestamp(int(e[0])).strftime('%Y-%m-%d %H:%M:%S'),e) for e in d]
    #print(result)
    pickle.dump( result, open("./data/files/eth/"+pfilename, "wb") )
    #end = start
    #start = (dateutil.parser.parse(end) - timedelta(hours=0, minutes=60)).isoformat()
    time.sleep(1)
  else:
    print("Skipping", pfilename, "since exists")

  end = start
  start = (dateutil.parser.parse(end) - timedelta(hours=0, minutes=60)).isoformat()