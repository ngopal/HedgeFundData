import urllib.request
import json
import sys
import pprint, time

# Obtain data from GET
def get_call(ticker):
    """ ticker should be all caps ticker symbol"""
    url = 'https://api.stocktwits.com/api/2/streams/symbol/'+str(ticker)+'.json'
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    result = response.read()
    res = json.loads(result)
    return res

def extract_relevant_fields(res):
    """Extract relevant fields"""
    filtered_res = []
    for i in res["messages"]:
        if i["entities"]["sentiment"]:
            filtered_res.append((i["created_at"],i["entities"]["sentiment"]["basic"],i["user"]["like_count"],i["user"]["username"]))
        else:
            filtered_res.append((i["created_at"],'Neutral',i["user"]["like_count"],i["user"]["username"]))
    return filtered_res

pprint.PrettyPrinter(indent=4)

mem = []
while True:
    vals = extract_relevant_fields(get_call("TSLA"))
    for v in vals:
        if v in mem:
            pass
        else:
            mem.append(v)
    records = set(mem)
    pprint.pprint(records)
    print(len(records))
    # if new records, then append new records to txt file
    time.sleep(30)



