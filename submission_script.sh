#!/bin/bash
 
PARTITION=$1
SERIESNAME=$2
CONSTRAINT=$3
OUTPUT=$4
ERROR=$5
UNITS=$6
SUBMISSIONS=$7
CONTINUE=${8}
DEPTH=${9}
AUG=${10}
MPARAMS=${11}
POSITION=${12}
FSMULT=${13}
KDIV=${14}
AUXWEIGHT=${15}
USECASE=${16}
PREFIX=${17}



if [[ "$CONTINUE" == "continue" ]]
then
    CONTINUE=1
else
    CONTINUE=0
fi


sbatch -J "$SERIESNAME" -d singleton --exclude=gpu-g30,gpu-g1 --partition="$PARTITION" --constraint="$CONSTRAINT" --output="$OUTPUT" --error="$ERROR" -c"$UNITS" run.sh -d $DEPTH -a "$AUG" -m "$MPARAMS" -p "$POSITION" -f $FSMULT -k $KDIV -w "$AUXWEIGHT" -u "$USECASE" -c $CONTINUE -s "$PREFIX"

END=$SUBMISSIONS
CONTINUE=1
for ((i=1; i<END; i++)); do
    sbatch -J "$SERIESNAME" -d singleton --exclude=gpu-g30,gpu-g1 --partition="$PARTITION" --constraint="$CONSTRAINT" --output="$OUTPUT" --error="$ERROR" -c"$UNITS" run.sh -d $DEPTH -a "$AUG" -m "$MPARAMS" -p "$POSITION" -f $FSMULT -k $KDIV -w "$AUXWEIGHT" -u "$USECASE" -c $CONTINUE -s "$PREFIX"
done
