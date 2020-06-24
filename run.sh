#!/bin/bash
 
while getopts ":d:a:m:p:f:k:w:c:u:s:" opt; do
  case $opt in
    d) pdepth=$OPTARG
    ;;
    a) paug="$OPTARG"
    ;;
    m) pmparams="$OPTARG"
    ;;
    p) pposition="$OPTARG"
    ;;
    f) pfsmult=$OPTARG
    ;;
    k) pkdiv=$OPTARG
    ;;
    w) pauxweight=$OPTARG
    ;;
    c) pcontinue=$OPTARG
    ;;
    u) pusecase=$OPTARG
    ;;
    s) pprefix=$OPTARG
    ;;
    \?) echo "Invalid option $OPTARG" >&2
    ;;
  esac
done


source ~/miniconda3/bin/activate baby


if [[ "$pcontinue" -eq "1" ]]
then
    python -m src -depth=$pdepth -aug="$paug" -mparams="$pmparams" -position="$pposition" -fsmult=$pfsmult -kdiv=$pkdiv -auxweight="$pauxweight" -usecase="$pusecase" -load=1 -prefix="$pprefix"
else
    python -m src -depth=$pdepth -aug="$paug" -mparams="$pmparams" -position="$pposition" -fsmult=$pfsmult -kdiv=$pkdiv -auxweight="$pauxweight" -usecase="$pusecase" -load=0 -prefix="$pprefix"
fi
