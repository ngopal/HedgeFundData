
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
" > ./reports/auto_report.html

cp ./reports/auto_report.html ./reports/auto_report_MAE.html;

myArray=( "$@" )

# Generate Text Files
for arg in "${myArray[@]}"; do
   echo "$arg"
   Rscript ./data/generators/X.Rscript $arg
   cat ./data/files/"$arg".csv | cut -d"," -f2 | sed '1d' > ./data/files/$arg"_open.csv"
   python forecast/simple_lstm/simple.py $arg 
   echo "
    <div class="row">
    <div class="column">
      <h2>$arg</h2>
      <embed src="${arg}.txt">
    </div>
    <div class="column">
      <img src="${arg}_performance.png" alt="Performance" style="width:100%">
    </div>
    <div class="column">
      <img src="${arg}_prediction.png" alt="Prediction" style="width:100%">
    </div>
  </div>
   " >> ./reports/auto_report.html
   python forecast/simple_lstm/simple_MAE.py $arg
   echo "
    <div class="row">
    <div class="column">
      <h2>$arg</h2>
      <embed src="${arg}_MAE.txt">
    </div>
    <div class="column">
      <img src="${arg}_performance_MAE.png" alt="Performance" style="width:100%">
    </div>
    <div class="column">
      <img src="${arg}_prediction_MAE.png" alt="Prediction" style="width:100%">
    </div>
  </div>
   " >> ./reports/auto_report_MAE.html
done

echo "</html>" >> ./reports/auto_report.html
echo "</html>" >> ./reports/auto_report_MAE.html
