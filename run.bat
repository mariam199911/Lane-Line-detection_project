set /p type_to_process="Do you want to process a video or an image [v/i]?"
set /p in_path="Enter a relative path to the input:"
set /p out_path="Enter a relative path to the output:"
set /p debug_mode="Do you want to enable debug mode [y/n]?"


python main.py %type_to_process% %in_path% %out_path% %debug_mode%