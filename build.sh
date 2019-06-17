echo "Starting Script"
Rscript ./data/generators/X.Multiple.Rscript $(cat tickers/tickers_all.txt)
wait;
echo "Generating Features and Building Model"
python MEDUSA_V1/medusa_v1.python
wait;
git add .
git commit -m "auto commit"
git push