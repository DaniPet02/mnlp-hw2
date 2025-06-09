#!/bin/bash
USERNAME="dpetrini"
LOCAL=$(realpath "./toSync/")
LOCAL="$LOCAL/"
REMOTE="/leonardo/home/userexternal/$USERNAME"

rsync -avz --progress $LOCAL $USERNAME@data.leonardo.cineca.it:$REMOTE