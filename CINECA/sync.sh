#!/bin/bash
USERNAME="apizzi00"
LOCAL="./toSync/"
REMOTE="/leonardo/home/userexternal/$USERNAME"

rsync -avz $LOCAL $USERNAME@data.leonardo.cineca.it:$REMOTE
