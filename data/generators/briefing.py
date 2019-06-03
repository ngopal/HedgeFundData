import requests
from bs4 import BeautifulSoup
import time, random, pickle, os, datetime


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# For Each Day
def getStoriesForDay(day):
    day = requests.get(day, headers = headers)
    soup = BeautifulSoup(day.text)
    stories = soup.findAll("a", {"class": "cell-story-title"})
    return([(i.text, "https://www.briefing.com/"+i["href"]) for i in stories])

#   For Each Update
def getMarketUpdate(link):
    waittime = random.randrange(1,5)
    print("UPDATE WAIT", waittime)
    time.sleep(waittime)
    day_update = requests.get(link, headers = headers)
    soup = BeautifulSoup(day_update.text)
    date_of_story = paragraphs = soup.find("div", {"class": "u-date"})
    paragraphs = soup.findAll("div", {"class": "padding-top10 padding-bottom5"})
    market_snapshot = soup.find("table", {"class" : "smu-table"})
    data = []
    try:
        rows = market_snapshot.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele]) # Get rid of empty values
        return( (date_of_story.text, [i.text for i in paragraphs], data))
    except:
        print("FAILED")
        return( (-1, [-1], []))

# Tests
# print(getMarketUpdate("https://www.briefing.com/investor/markets/stock-market-update/2012/1/20/stocks-squeek-out-slight-gain.htm"))
# print(getStoriesForDay("https://www.briefing.com/investor/markets/stock-market-update/2012/1/20/"))
# print(getStoriesForDay("https://www.briefing.com/investor/markets/stock-market-update/2012/1/22"))

def getAllDataForDay(date_of_incident):
    storydata =[]
    for story in getStoriesForDay(date_of_incident):
        entry = {}
        entry["headline"] = story[0]
        entry["url"] = story[1]
        waittime = random.randrange(1,13)
        print(entry)
        print("INTRA WAIT", waittime)
        time.sleep(waittime)
        res = getMarketUpdate(story[1])
        entry["date"] = res[0]
        entry["text"] = res[1]
        entry["tickerinfo"] = res[2]
        print(entry)
        storydata.append(entry)
    return(storydata)

now = datetime.datetime.now()
for y in range(2016,2020):
    for m in range(1,13):
        for d in range(1,32):
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
            if str(y)+"_"+str(m)+"_"+str(d)+".p" not in os.listdir("./briefings/"):
                date_url = "https://www.briefing.com/investor/markets/stock-market-update/"+str(y)+"/"+str(m)+"/"+str(d)+"/"
                result = getAllDataForDay(date_url)
                pickle.dump( result, open( "./briefings/"+str(y)+"_"+str(m)+"_"+str(d)+".p", "wb" ) )

print("DONE")