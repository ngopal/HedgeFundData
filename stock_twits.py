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
bull_count = 0
bear_count = 0
neutral_count = 0
old_records = set(mem)
while True:
    vals = extract_relevant_fields(get_call(sys.argv[1].decode('utf-8')))
    for v in vals:
        if v in mem:
            pass
        else:
            mem.append(v)
            if v[1] == 'Bearish':
                bear_count += 1
            elif v[1] == 'Bullish':
                bull_count += 1
            else:
                neutral_count += 1
    records = set(mem)
    pprint.pprint(records)
    print(len(records))
    try:
        print(bear_count, neutral_count, bull_count, bull_count/bear_count)
    except:
        pass
    # if new records, then append new records to txt file
    f = open('./sentiment'+sys.argv[1]+'.txt', 'a')
    for e in records.difference(old_records):
        f.write(e[0]+'\t'+e[1]+'\t'+str(e[2])+'\t'+e[3]+'\n')
    f.close()
    print("NEW ENTRIES", records.difference(old_records))
    time.sleep(30)
    old_records = records



