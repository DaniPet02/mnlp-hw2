#!/bin/bash

LOCAL="./toSync"
REMOTE="/leonardo/home/userexternal/dpetrini"
scp -r $LOCAL dpetrini@data.leonardo.cineca.it:$REMOTE

