myArray=( "$@" )

# Generate Text Files
for arg in "${myArray[@]}"; do
   echo "$arg"
   Rscript ./generators/X.Rscript $arg
   cat ./files/"$arg".csv | cut -d"," -f2 | sed '1d' > ./files/$arg"_open.csv"
done

