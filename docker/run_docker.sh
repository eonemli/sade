#!/bin/bash

NAME="sade"
PORT=9990
CODESPACE=/GROND_STOR/amahmood/workspace/
#CODESPACE=/ASD/ahsan_projects/test/

docker stop $NAME-docker || true # Exits gracefully if container doesnt exist

docker run \
	-d \
	--rm \
	--init \
	--name $NAME-docker \
	--ipc=host \
	-e JUPYTER_ENABLE_LAB=yes \
	--gpus device=all \
	--mount type=bind,src=/ASD/ahsan_projects/braintypicality/workdir/,target=/workdir/ \
	--mount type=bind,src="/BEE/Connectome/ABCD/",target=/DATA \
	--mount type=bind,src="/ASD/",target=/ASD \
	--mount type=bind,src="/UTexas",target=/UTexas \
	--mount type=bind,src=$CODESPACE,target=/codespace \
	-p $PORT:8888 \
	-p 6006:6006 \
	$USER/pytorch_sde:latest \
	jupyter lab --ip 0.0.0.0 --notebook-dir=/ --no-browser --allow-root
