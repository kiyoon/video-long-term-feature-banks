#!/bin/bash

if [ $# -lt 2 ]
then
	echo "usage: $0 [input_dir (should be .../frames)] [train.csv or val.csv]"
	echo "It will print all the missing files"
	exit 1
fi

input_dir="$1"
trainval_csv="$2"

trainval=$(cat "$trainval_csv" | sed 1d)
num_files=$(echo "$trainval" | wc -l)
id=1

#bash_start_time=$(date +%s.%N)

while read line
do
	filename=$(echo "$line" | awk '{print $4}')

	>&2 echo ${id} / $num_files
	
	if [ ! -f "$input_dir/$filename" ]; then
	    echo "$input_dir/$filename"
	fi

	#bash_end_time=$(date +%s.%N)
	#time_diff=$(echo "$bash_end_time - $bash_start_time" | bc)
	#average_time=$(echo "$time_diff / ($id+1)" | bc -l)
	#echo "average processing time per file: $average_time"
	(( id++ ))
done <<< "$trainval"
