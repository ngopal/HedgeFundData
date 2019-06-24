#!/bin/bash
for arg in "$@"
do
    if [ "$arg" == "--data" ] || [ "$arg" == "-d" ]
    then
        echo "Downloading Data"
        Rscript ./data/generators/X.Multiple.Rscript $(cat tickers/tickers_all.txt)
        wait;
    fi
    if [ "$arg" == "--model" ] || [ "$arg" == "-m" ]
    then
        echo "Generating Features and Building Model"
        python MEDUSA_V1/medusa_v1.py
        wait;
    fi
    if [ "$arg" == "--inference-only" ] || [ "$arg" == "-i" ]
    then
        echo "Generating Features and Building Model"
        python MEDUSA_V1/medusa_v1.py --inference-only 1
        wait;
    fi
    if [ "$arg" == "--push" ] || [ "$arg" == "-p" ]
    then
        git add reports/*
        wait;
        git add data/files/*
        wait;
        git commit -m "auto commit"
        wait;
        git push
    fi
    echo "DONE."
done


