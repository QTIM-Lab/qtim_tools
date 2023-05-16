
from qtim_tools.test_data import load_test_file

# test_file = "your_file_here.nii.gz"
test_file = load_test_file(data="dce_tofts_v6")
print('Test File Loaded: ' + test_file)

from qtim_tools.qtim_dce import tofts_parametric_mapper

# Input Parameters
T1_tissue=1000
T1_blood=1440
relaxivity=.0045
TR=5
TE=2.1 
scan_time_seconds=(11*60) 
hematocrit=0.45
injection_start_time_seconds=60 
flip_angle_degrees=30

test_file_label = load_test_file(data="dce_tofts_v6_label")
print('Loaded test label: ' + test_file_label)

# Label File Parameters
label_file=test_file_label

# AIF Parameters
AIF_label_file=[]
AIF_text_file=[]
AIF_mode="population" # options: population, label, textfile

# T1 Map Parameters
T1_map_file=[]

# Output and Processing Parameters.
processes = 2;
output_filepath_prefix = "./Example_Breast_Data_"

tofts_parametric_mapper.execute(input_filepath=test_file,
T1_tissue=T1_tissue,
T1_blood=T1_blood,
relaxivity=relaxivity,
TR=TR, 
TE=TE, 
scan_time_seconds=scan_time_seconds, 
hematocrit=hematocrit, 
injection_start_time_seconds=injection_start_time_seconds, 
flip_angle_degrees=flip_angle_degrees, 
label_file=test_file_label,
T1_map_file=[],
AIF_label_file=[],
AIF_text_file=[],
AIF_mode=AIF_mode,
outfile_prefix=output_filepath_prefix, 
processes=processes)






