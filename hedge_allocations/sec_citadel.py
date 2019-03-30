# Vital note on the limitations of this document: http://csinvesting.org/2012/05/16/lessons-on-reading-a-13-f/

import feedparser

# Get Filing Link
d = feedparser.parse('https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001423053&CIK=0001423053&type=13F-HR%25&dateb=&owner=exclude&start=0&count=40&output=atom')

filing_link = d['entries'][0]['link']

# Parse Filing Link Page and Extract Link to XML
from bs4 import BeautifulSoup
import requests

page = requests.get(filing_link)
soup = BeautifulSoup(page.content, 'html.parser')
filing_date = str(soup.findAll("div", {"class": "info"})[0].text)
print([k for k in [l.get("href") for l in soup.find_all("a")] if "Table" in k and "xsl" in k][0])
xml_link = "https://www.sec.gov" + [k for k in [l.get("href") for l in soup.find_all("a")] if "Table" in k and "xsl" in k][0]

# Parse Investment Doc
from collections import defaultdict
ip = requests.get(xml_link)
soup = BeautifulSoup(ip.content, 'html.parser')

investments = defaultdict(int)
for row in soup.find_all("tr"):
    #print(row)
    row = [r.text for r in row.find_all("td")]
    try:
        investments[row[0]] = int(row[3].replace(",",''))
    except:
        #print("pass", row)
        pass

 # List top 10 investments
print("Filing Date: " + filing_date)
for k,v in sorted(investments.items(), key=lambda x: x[1], reverse=True)[0:10]:
    print(k+":"+str(v))
