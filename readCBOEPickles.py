import pickle
import os
from bs4 import BeautifulSoup
import pandas as pd
import datetime

# f = open("./cboe/4_1_2016_cboe_futures.dat", "rb")
# d = pickle.loads(f.read())

# table1_data = []
# table2_data = []

def GenerateTable1(date):
    filename = "./cboe/"+date+"_cboe_futures.p"
    # if len([w for w in os.listdir("./cboe/") if w == date+"_cboe_futures.dat"]) == 0:
    #     return -1
    f = open(filename, "rb")
    d = pickle.loads(f.read())
    table1_data = []
    for i in d:
        t = d[1].replace("\r","").replace("&nbsp","").split("\n")
        for k, v in enumerate(t):
            if '(' in v and ')' in v:
                try:
                    table1 = {"date" : datetime.datetime(month=int(date.split("_")[1]), day=int(date.split("_")[0]), year=int(date.split("_")[2])).isoformat().split("T")[0], "title" : '', "vol_call" : None, "vol_put" : None, "vol_total" : None, "oi_call" : None, "oi_put" : None, "oi_total" : None}
                    row = t[k:(k+37)]
                    row = [j for j in row if '\xa0' not in j]
                    row = [j for j in row if len(j) > 0]
                    table1["title"] = (lambda x: x if ':' not in x else WEEKLY)(row[0].split("(")[1].split(")")[0])
                    if not row[5].isdigit():
                        continue
                    table1["vol_call"] = row[5].replace(",", "")
                    table1["vol_put"] = row[6].replace(",", "")
                    table1["vol_total"] = row[7].replace(",", "")
                    table1["oi_call"] = row[9].replace(",", "")
                    table1["oi_put"] = row[10].replace(",", "")
                    table1["oi_total"] = row[11].replace(",", "")
                    table1_data.append(table1)
                except:
                    pass
    return pd.DataFrame(table1_data)

def GenerateTable2(date):
    filename = "./cboe/"+date+"_cboe_futures.p"
    # if len([w for w in os.listdir("./cboe/") if w == date+"_cboe_futures.dat"]) == 0:
    #     return -1
    f = open(filename, "rb")
    d = pickle.loads(f.read())
    table2_data = []
    for i in d:
        t = d[1].replace("\r","").replace("&nbsp","").split("\n")
        for k, v in enumerate(t):
            if '(' in v and ')' in v:
                try:
                    table2 = {"date" : datetime.datetime(month=int(date.split("_")[1]), day=int(date.split("_")[0]), year=int(date.split("_")[2])).isoformat().split("T")[0], "title" : '', "level_high" : None, "level_low" : None, "level_close" : None, "level_change" : None}
                    row = t[k:(k+37)]
                    row = [j for j in row if '\xa0' not in j]
                    row = [j for j in row if len(j) > 0]
                    if not row[5].isdigit():
                        continue
                    if row[12] == "High":
                        if len(row[17:]) > 0:
                            table2["title"] = (lambda x: x if ':' not in x else WEEKLY)(row[0].split("(")[1].split(")")[0])
                            table2["level_high"] = row[17].replace(",", "")
                            table2["level_low"] = row[18].replace(",", "")
                            table2["level_close"] = row[19].replace(",", "")
                            table2["level_change"] = row[20].replace(",", "")
                    else:
                        pass
                    table2_data.append(table2)
                except:
                    pass
    return pd.DataFrame(table2_data)

# print(GenerateTable1("26_11_2018"))
# print(GenerateTable2("4_1_2016"))

dfs1 = []
dfs2 = []
for f in os.listdir("./cboe/"):
    print(f)
    print(f.split("_cboe")[0])
    dfs1.append(GenerateTable1(f.split("_cboe")[0]))
    dfs2.append(GenerateTable2(f.split("_cboe")[0]))

combined_df1 = pd.concat(dfs1)
combined_df1 = combined_df1.set_index("date")
print(combined_df1.shape)
combined_df1.to_pickle("./data/files/cboedf1.p")
print("Wrote df1 to file")

combined_df2 = pd.concat(dfs2)
combined_df2 = combined_df2.set_index("date")
print(combined_df2.shape)
combined_df2.to_pickle("./data/files/cboedf2.p")
print("Wrote df1 to file")
# loaded_df = pd.read_pickle("./data/files/cboedf.p")
# print(loaded_df)
# print(loaded_df.shape)
