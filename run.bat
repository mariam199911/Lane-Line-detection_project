# Take run arguments
echo "Do you want to process a video or an image [v/i]?"
read type_to_process
echo "Enter a relative path to the input:"
read in_path
echo "Enter a relative path to the output:"
read out_path
echo "Do you want to enable debug mode [y/n]?"
read debug_mode

# run the python file
python3 main.py $type_to_process $in_path $out_path $debug_mode