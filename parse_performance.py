import sys, os

REPORT_DIR = "./reports"
files = os.listdir(REPORT_DIR)
files = [f for f in files if ".txt" in f]

output = open('portfolio_data.csv', "w")

for f in files:
    fo = open(REPORT_DIR+os.sep+f, "r")
    ticker = f.split(".")[0].strip()
    lines = [l for l in fo.readlines()]
    val_loss = lines[1].split(" ")[1].strip()
    slope = lines[-1].split(" ")[1].strip()
    print(ticker, val_loss, slope)
    output.write(ticker+','+val_loss+','+slope+'\n')
    fo.close()
output.close()