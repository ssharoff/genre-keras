evalclass.py $1 $2 >$2.out
paste $2.out $1 | sort | uniq -c | sort -nrs >$3
./precision.sh A8 $3
#A1 is special
CORR=`grep -w 'A1' $3 | sed 's/$/ /' | egrep '__A1 ' | gawk '{print($1)}' | sumcolumn `
TOT=`grep -w 'A1' $3 | gawk '{print($1)}' | sumcolumn `
echo A1 =$CORR/$TOT
./precision.sh A17 $3
./precision.sh A16 $3
./precision.sh A7 $3
./precision.sh A14 $3
./precision.sh A9 $3
./precision.sh A4 $3
./precision.sh A11 $3
./precision.sh A12 $3
