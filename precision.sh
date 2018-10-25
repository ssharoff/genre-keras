A=$1
CORR=`grep ' '$A $2 | grep __$A | gawk '{print($1)}' | sumcolumn `
TOT=`grep ' '$A $2  | gawk '{print($1)}' | sumcolumn`
echo $A  =$CORR/$TOT
