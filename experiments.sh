#!/bin/bash

addition=""



# --------------- Hierarchical Sparse Layer Insertion -----------------

if false; then
    addition="arch_"

    PREFIX="ArchInsertion"

    DEPTH=6
    AUG="true"
    MPARAMS="true"
    NUMITER=7
    CONTINUE="continue"

    FSMULT=4
    AUXWEIGHT=0.7


    for KDIV in 1 2 4; do
        for POSITION in "First_Hierarchical" "01_Hierarchical" "012_Hierarchical" "0123_Hierarchical" "01234_Hierarchical" "012345_Hierarchical"; do
            for USECASE in "supervise" "random" "pretrain" "regularize"; do
                TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv6_Sparse";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
            done
        done
    done
fi


#--------------------- Conv12 Extension ---------------------------------


if false; then
    addition="12_"

    PREFIX="NEW12_Insertion"

    DEPTH=12
    NUMITER=7
    CONTINUE="continue"

    FSMULT=4
    AUXWEIGHT=0.7


    for KDIV in 1 2 4; do
        for POSITION in "0" "01" "012" "0123" "01234" "012345"; do
	        for USECASE in "random" "pretrain" "supervise" "regularize"; do
                TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv12";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "12g" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "true" "true" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
            done
        done
    done
fi


# --------------- Resnet Extension --------------------------------------

if false; then
    addition="res_"

    PREFIX="12Res"

    DEPTH=12
    AUG="true"
    MPARAMS="true"
    NUMITER=7
    CONTINUE="continue"

    FSMULT=4
    AUXWEIGHT=0.7

    for KDIV in 1 2 4; do
        for POSITION in "0_Res" "01_Res" "012_Res" "0123_Res" "01234_Res" "012345_Res"; do
            for USECASE in "supervise" "random" "pretrain" "regularize"; do
                TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv12Res";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "11g" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
            done
        done
    done
fi


#  --------------- Primary / Aux loss weighting search ------------------

if true; then

    addition="hyper_"

    PREFIX="HYPERSEARCH"

    DEPTH=6
    KDIV=1
    AUG="true"
    MPARAMS="true"
    NUMITER=3
    CONTINUE="continue"

    for FSMULT in 2 4; do
        for POSITION in "First" "012" "012345"; do
            for AUXWEIGHT in 0.9 0.7 0.5 0.3 0.1; do
                for LR in 0.2 0.1 0.03 0.01 0.005; do
                    TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Lr:${LR}_Conv6HyperSearch";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}" ${LR};
                done
            done
        done
    done

fi


# --------------- Sparse Layer Insertion -----------------

if false; then
    addition="oldways"

    PREFIX="OldWaySearch"

    DEPTH=6
    AUG="true"
    MPARAMS="true"
    NUMITER=5
    CONTINUE="continue"

    FSMULT=4
    AUXWEIGHT=0.7


    #for KDIV in 1 2 4; do
        KDIV=2
        #for POSITION in "First" "01" "012" "0123" "01234" "012345"; do
            POSITION="012345"
            for USECASE in "supervise" "random" "pretrain" "regularize"; do
                TAG="${addition}Use:${USECASE}_Aux:${AUXWEIGHT}_FS:${FSMULT}_KD:${KDIV}_Pos:${POSITION}_Conv6_Sparse";
                ./submission_script.sh mmaire-gpu "${TAG}Series" "" "log/${TAG}_std.out" "log/${TAG}_std.err" 1 ${NUMITER} "${CONTINUE}" ${DEPTH} "${AUG}" "${MPARAMS}" "${POSITION}" ${FSMULT} ${KDIV} "${AUXWEIGHT}" "${USECASE}" "${PREFIX}";
            done
        #done
    #done
fi
 




