
python sort_file.py all_english.txt all_french.txt
python sort_file.py validation_english.txt validation_french.txt
python integerize.py
python pad.py output.txt
python pad.py output_validation.txt
mv output.txt.padded train.txt
mv output_validation.txt validation.txt
cp train.txt single_layer_gpu_google_model/
cp validation.txt single_layer_gpu_google_model/