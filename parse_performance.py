import sys, os

REPORT_DIR = "./reports"
files = os.listdir(REPORT_DIR)
files = [f for f in files if ".txt" in f]

for f in files:
    fo = open(REPORT_DIR+os.sep+f, "r")
    print(f, [l for l in fo.readlines()])
    fo.close()