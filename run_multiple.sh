
echo "<html>
<style>
/* Three image containers (use 25% for four, and 50% for two, etc) */
.column {
  float: left;
  width: 30%;
  padding: 5px;
}

/* Clear floats after image containers */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style
<body>
" > ./reports/auto_report_MULTI.html

cp ./reports/auto_report_MULTI.html ./reports/auto_report_MULTI_MAE.html

myArray=( "$@" )

# Generate Text Files
for arg in "${myArray[@]}"; do
   echo "$arg"
#    Rscript ./data/generators/X.Rscript $arg # needs to be X.Multiple.R
   python forecast/simple_lstm/simple_all.py $arg 
   echo "
    <div class="row">
    <div class="column">
      <h2>$arg</h2>
      <embed src="${arg}.txt">
    </div>
    <div class="column">
      <img src="multiple_concatenated_tickers_${arg}_performance.png" alt="Performance" style="width:100%">
    </div>
    <div class="column">
      <img src="multiple_concatenated_tickers_${arg}_prediction.png" alt="Prediction" style="width:100%">
    </div>
  </div>
   " >> ./reports/auto_report_MULTI.html
   python forecast/simple_lstm/simple_all_MAE.py $arg 
   echo "
    <div class="row">
    <div class="column">
      <h2>$arg</h2>
      <embed src="${arg}.txt">
    </div>
    <div class="column">
      <img src="multiple_concatenated_tickers_${arg}_performance_MAE.png" alt="Performance" style="width:100%">
    </div>
    <div class="column">
      <img src="multiple_concatenated_tickers_${arg}_prediction_MAE.png" alt="Prediction" style="width:100%">
    </div>
  </div>
   " >> ./reports/auto_report_MULTI_MAE.html
done

echo "</html>" >> ./reports/auto_report_MULTI.html
echo "</html>" >> ./reports/auto_report_MULTI_MAE.html
