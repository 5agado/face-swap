#!/usr/bin/env bash

FRAMERATE=25

WORKDIR=""
INIT_FOLDER=false
EXTRACT_FACES=false
SWAP=false
STEP_MOD=1

while getopts 'w:iesf:t:' flag; do
  case "${flag}" in
    w) WORKDIR="${OPTARG}"
       ;;
    i) INIT_FOLDER=true
       ;;
    e) EXTRACT_FACES=true
       ;;
    s) SWAP=true
       ;;
    f) FACE_DIRS+=("$OPTARG")
       ;;
    t) STEP_MOD="${OPTARG}"
  esac
done

if [ -z ${WORKDIR} ]
then
	echo "Usage: ./scripts.sh -w <work_dir> [-f]"
	exit 1
fi


# if required, setup folder and extract frames
if [ ${INIT_FOLDER} = true ]
then
    mkdir -p ${WORKDIR}/{faces,out,test_out_gifs}

    # Extract all frames from video. Not needed if we operate directly at video level
    # *.{gif,webm} doesn't work, seems to fail if not both of the extensions are covered
    #for ext in gif webm; do
    #    ffmpeg -hide_banner -loglevel panic -i ${WORKDIR}/*.${ext} ${WORKDIR}/frames/frame_%04d.png
    #done
    # scale frames
    #WIDTH=480
    #ffmpeg -hide_banner -loglevel panic -i ${WORKDIR}/*.gif -filter:v "scale=${WIDTH}:-1" ${WORKDIR}/frames/frame_%04d.png
    #ffmpeg -hide_banner -loglevel panic -i ${WORKDIR}/*.webm -filter:v "scale=${WIDTH}:-1" ${WORKDIR}/frames/frame_%04d.png
fi

# if required, extract faces
if [ ${EXTRACT_FACES} = true ]
then
    for ext in gif webm mp4; do
        extract -i ${WORKDIR}/*.${ext} -o ${WORKDIR}/faces/ -s ${STEP_MOD}
    done
fi

# if required, swap faces and create gif from frames
if [ ${SWAP} = true ]
then
    for ext in gif webm mp4; do
        deep_swap -i ${WORKDIR}/*.${ext} -o ${WORKDIR}/test_out_gifs/out_$(date +%s).mp4 -A -model_name masked_gan -model_version v1
    done
    #deep_swap -i ${WORKDIR}/frames/ -o ${WORKDIR}/out/ -A -model_name masked_gan -model_version v1 -process_images
    #ffmpeg -hide_banner -framerate ${FRAMERATE} -i ${WORKDIR}/out/frame_%04d.png ${WORKDIR}/test_out_gifs/out_$(date +%s).gif
    exit 0
fi

# if face folders have been provided, copy from all in the workdir
if [ ${FACE_DIRS} ]
then
    mkdir -p ${WORKDIR}/train_faces/
    num=0
    for dir in "${FACE_DIRS[@]}"; do
        echo " - $dir"
        for i in ${dir}/faces/*.jpg; do
            let num=num+1
            if [ "$(($num % $STEP_MOD))" -eq 0 ]
            then cp ${i} ${WORKDIR}/train_faces/${num}.jpg
            fi
        done
    done
fi