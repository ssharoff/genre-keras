#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -m e
#x$ -pe smp 13
#x$ -l h_vmem=40G
#$ -l coproc_v100=1

# 1 language (num/vec)
# 2 training.ol training.csv
# 3 else

module add anaconda
module add mkl
#module add cuda

export LD_LIBRARY_PATH+=:/nobackup/smlss/tf/lib

source activate /nobackup/smlss/tf

echo $*
echo    classifier.py -d $1.brieftag.num -f 6000 -x 500 -e 15 -c 10 --valsplit 0.1 -v 2 -g 0 -1 $1.vec.xz -i $2.ol -a $2.csv $3
python3 classifier.py -d $1.brieftag.num -f 6000 -x 500 -e 15 -c 10 --valsplit 0.1 -v 2 -g 0 -1 $1.vec.xz -i $2.ol -a $2.csv $3
#precisions.sh $2.ol.gold classifier.py_-d_$2.brieftag.num*iter1.vec*ru.ol.pred $2$1.pred.num
