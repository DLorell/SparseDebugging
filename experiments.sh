#!/bin/bash

addition=""



#--------------------- Conv12 Extension ---------------------------------


if true; then
    addition="12_"

    PREFIX="12_Insertion"

    DEPTH=12
    NUMITER=5
    CONTINUE="continue"

    FSMULT=4
    AUXWEIGHT=0.7


    for KDIV in 1 2 4; do
        for POSITION in "0" "01" "012" "0123" "01234" "012345"; do
	        for USECASE in "random" "pretrain" "supervise" "regularize"; do
                TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv12";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "true" "true" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
            done
        done
    done
fi


# --------------- Resnet Extension --------------------------------------

if false; then
    addition="res_"

    PREFIX="NewResnet"

    DEPTH=6
    AUG="true"
    MPARAMS="true"
    NUMITER=5
    CONTINUE="continue"

    FSMULT=4
    AUXWEIGHT=0.7

    for KDIV in 1 2 4; do
        POSITION="Resnet_Sparse"
        #for POSITION in "First" "01" "012" "0123" "01234" "012345"; do
            for USECASE in "supervise" "random" "pretrain" "regularize"; do
                TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv6_Sparse";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "12g" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
            done
        #done
    done
fi




#  --------------- Primary / Aux loss weighting search ------------------

if false; then

    PREFIX=""

    DEPTH=6
    KDIV=1
    AUG="true"
    MPARAMS="true"
    NUMITER=2
    CONTINUE="continue"

    for FSMULT in 2 4; do
        for POSITION in "First" "012" "012345"; do
            for AUXWEIGHT in 0.1 0.25 0.5 0.6 0.7 0.8 0.9 0.99; do
                TAG="${addition}Aux:${AUXWEIGHT}_FS:${FSMULT}_Pos:${POSITION}_Conv6_Sparse";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "regularize" "${PREFIX}";
            done
        done
    done

fi


# --------------- Sparse Layer Insertion -----------------

if false; then
    addition="premp"

    PREFIX="Insertion"

    DEPTH=6
    AUG="true"
    MPARAMS="true"
    NUMITER=5
    CONTINUE="continue"

    FSMULT=4
    AUXWEIGHT=0.7


    KDIV=2
    POSITION="012"
    for USECASE in "pretrain" "regularize"; do
        TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv6_Sparse";
        ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
    done

    POSITION="0123"
    USECASE="regularize"
    TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv6_Sparse";
    ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
    

    #for KDIV in 1 2 4; do
    #    for POSITION in "First" "01" "012" "0123" "01234" "012345"; do
    #        for USECASE in "supervise" "random" "pretrain" "regularize"; do
    #            TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv6_Sparse";
    #            ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
    #        done
    #    done
    #done
fi
 




