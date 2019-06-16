echo "Starting Script"
Rscript ./data/generators/X.Multiple.Rscript $(cat tickers/tickers_all.txt)
wait;
echo "Generating Features and Building Model"
python 