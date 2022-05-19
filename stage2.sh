type_to_process=$1
in_path=$2
out_path=$3
debug_mode=$4

# run the python file
python3 stage2.py $type_to_process $in_path $out_path $debug_mode