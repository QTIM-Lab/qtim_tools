import argparse
import sys

import qtim_tools


class qtim_commands(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='A number of pre-packaged command used by the Quantiative Tumor Imaging Lab at the Martinos Center',
            usage='''qtim <command> [<args>]

The following commands are available:
   coregistration               Coregister data to one volume, or through a series of steps.
   deep_learning_preprocess     Preprocess coregistered volumes for deep learning analysis.
   deep_learning_experiment     Copy DL files into test/train folders for an experiment.
   dti_conversion               Generate DTI volumes from RAWDATA.
   label_statistics             Generate label statistics from the COREGISTRATION folder.
   mosaic                       Generate a mosaic from an input volume and optional label.
   dce_maps                     Calculates DCE parameters maps (e.g. ktrans, ve, auc) for a given 4D Nifti file.
''')

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Sorry, that\'s not one of the commands.')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def coregistration(self):
        parser = argparse.ArgumentParser(
            description='Coregister QTIM volumes to one volume, or according to a series of steps in a configuration file.')

        parser.add_argument('study_name', type=str)
        parser.add_argument('base_directory', type=str)
        parser.add_argument('-destination_volume', type=str)
        parser.add_argument('-config', type=str)   

        args = parser.parse_args(sys.argv[2:])
        print('Coregistering volumes for study directory... %s' % args.study_name)

        qtim_tools.qtim_pipelines.coregistration.coregister_pipeline(args.study_name, args.base_directory)

    def deep_learning_preprocess(self):
        parser = argparse.ArgumentParser(
            description='Prepare a study for a deep learning experiment. Performs N4 bias correction, skull-stripping, isotropic resampling, and zero-mean normalization.')

        parser.add_argument('study_name', type=str)
        parser.add_argument('base_directory', type=str)
        parser.add_argument('-skull_strip_label', type=str)
        parser.add_argument('-config', type=str)   

        args = parser.parse_args(sys.argv[2:])
        print('Preprocessing for deep learning in study directory... %s' % args.study_name)

        qtim_tools.qtim_pipelines.deep_learning.deep_learning_preprocess(args.study_name, args.base_directory)

    def deep_learning_experiment(self):
        parser = argparse.ArgumentParser(
            description='Prepare a study for a deep learning experiment. Performs N4 bias correction, skull-stripping, isotropic resampling, and zero-mean normalization.')

        parser.add_argument('base_directory', type=str) 
        parser.add_argument('output_directory', type=str)
        parser.add_argument('-config_file', type=str)

        args = parser.parse_args(sys.argv[2:])
        print('Creating a deep learning experiment in study directory... %s' % args.output_directory)

        qtim_tools.qtim_pipelines.deep_learning.deep_learning_experiment(args.base_directory, args.output_directory, args.config_file)

    def dti_conversion(self):
        parser = argparse.ArgumentParser(
            description='Generate DTI modalities from QTIM study raw data.')

        parser.add_argument('study_name')
        parser.add_argument('base_directory')
        parser.add_argument('-case', default=None)     
        parser.add_argument('-output_modalities',default=[])
        parser.add_argument('-overwrite',default=[False],type=eval)            

        args = parser.parse_args(sys.argv[2:])
        print('Running DTI conversion on study directory... %s' % args.study_name)

        qtim_tools.qtim_pipelines.dti_conversion.qtim_dti_conversion(study_name=args.study_name,base_directory=args.base_directory, specific_case=args.case, output_modalities=args.output_modalities, overwrite=args.overwrite)

    def label_statistics(self):
        parser = argparse.ArgumentParser(
            description='Generate label statistics from a QTIM study and a given ROI.')

        parser.add_argument('study_name', type=str)
        parser.add_argument('label_name', type=str)
        parser.add_argument('base_directory', type=str)
        parser.add_argument('-output_csv', type=str)
        parser.add_argument('-label_mode', type=str)
        parser.add_argument('-config', type=str)   

        args = parser.parse_args(sys.argv[2:])
        print('Running label statistics on study directory... %s' % args.study_name)

        qtim_tools.qtim_pipelines.label_statistics.qtim_study_statistics(args.study_name, args.label_name, args.base_directory, args.output_csv, args.label_mode)

    def mosaic(self):

        parser = argparse.ArgumentParser(
            description='Create a mosaic of 2D images from a 3D volume and, optionally, a 2D label.')

        parser.add_argument('input_volume', type=str)
        parser.add_argument('output_image', type=str)
        parser.add_argument('-label_volume', type=str)
        parser.add_argument('-step', type=int)

        args = parser.parse_args(sys.argv[2:])

        qtim_tools.qtim_visualization.image.create_mosaic(args.input_volume, args.output_image, args.label_volume, step=args.step)

    def radiomics(self):

        parser = argparse.ArgumentParser(
            description='Create a mosaic of 2D images from a 3D volume and, optionally, a 2D label.')

        parser.add_argument('input_volume', type=str)
        parser.add_argument('output_csv', type=str)
        parser.add_argument('-input_label', type=str)
        parser.add_argument('-levels', type=int, nargs='?', default=255, const=255)
        parser.add_argument('-normalize_intensities', action='store_true')

        args = parser.parse_args(sys.argv[2:])

        qtim_tools.qtim_features.generate_feature_list(args.input_volume, label_file=args.input_label, outfile=args.output_csv, levels=args.levels, normalize_intensities=args.normalize_intensities)
    
    def dce_maps(self):
        parser = argparse.ArgumentParser(description='calculates DCE parameters maps (e.g. ktrans, ve, auc) for a given 4D Nifti file.')
        parser.add_argument('--filepath',required=True)
        parser.add_argument('--T1_tissue',required=False,default=1000)
        parser.add_argument('--T1_blood',required=False,default=1440)
        parser.add_argument('--relaxivity',required=False,default=.0045)
        parser.add_argument('--TR',required=False,default=5)
        parser.add_argument('--TE',required=False,default=2.1)
        parser.add_argument('--scan_time_seconds',required=False,default=(11*60))
        parser.add_argument('--hematocrit',required=False,default=0.45)
        parser.add_argument('--injection_start_time_seconds',required=False,default=60)
        parser.add_argument('--flip_angle_degrees',required=False,default=30)
        parser.add_argument('--label_file',required=False,default=[])
        parser.add_argument('--label_suffix',required=False,default=[])
        parser.add_argument('--label_value',required=False,default=1)
        parser.add_argument('--mask_value',required=False,default=0)
        parser.add_argument('--mask_threshold',required=False,default=0)
        parser.add_argument('--T1_map_file',required=False,default=[])
        parser.add_argument('--T1_map_suffix',required=False,default=[])
        parser.add_argument('--AIF_label_file',required=False,default=[])
        parser.add_argument('--AIF_value_data',required=False,default=[])
        parser.add_argument('--AIF_value_suffix',required=False,default=[])
        parser.add_argument('--convert_AIF_values',required=False,default=True)
        parser.add_argument('--AIF_mode',required=False,default='population')
        parser.add_argument('--AIF_label_suffix',required=False,default=[])
        parser.add_argument('--AIF_label_value',required=False,default=1)
        parser.add_argument('--label_mode',required=False,default='separate')
        parser.add_argument('--param_file',required=False,default=[])
        parser.add_argument('--default_population_AIF',required=False,default=False)
        parser.add_argument('--initial_fitting_function_parameters',required=False,default=[.01,.01])
        parser.add_argument('--outputs',required=False,default=['ktrans'])
        parser.add_argument('--outfile_prefix',required=True)
        parser.add_argument('--processes',required=False,default=1)
        parser.add_argument('--gaussian_blur',required=False,default=.65)
        parser.add_argument('--gaussian_blur_axis',required=False,default=2)
        args=parser.parse_args(sys.argv[2:])

        
        print(f'Generating DCE maps from {args.filepath} to {args.outfile_prefix}')

        qtim_tools.qtim_dce.tofts_parametric_mapper.calc_DCE_properties_single(filepath=args.filepath, T1_tissue=args.T1_tissue, T1_blood=args.T1_blood, relaxivity=args.relaxivity, TR=args.TR, TE=args.TE, scan_time_seconds=args.scan_time_seconds, hematocrit=args.hematocrit, injection_start_time_seconds=args.injection_start_time_seconds, flip_angle_degrees=args.flip_angle_degrees, label_file=args.label_file, label_suffix=args.label_suffix, label_value=args.label_value, mask_value=args.mask_value, mask_threshold=args.mask_threshold, T1_map_file=args.T1_map_file, T1_map_suffix=args.T1_map_suffix, AIF_label_file=args.AIF_label_file,  AIF_value_data=args.AIF_value_data, AIF_value_suffix=args.AIF_value_suffix, convert_AIF_values=args.convert_AIF_values, AIF_mode=args.AIF_mode, AIF_label_suffix=args.AIF_label_suffix, AIF_label_value=args.AIF_label_value, label_mode=args.label_mode, param_file=args.param_file, default_population_AIF=args.default_population_AIF, initial_fitting_function_parameters=args.initial_fitting_function_parameters, outputs=args.outputs, outfile_prefix=args.outfile_prefix, processes=args.processes, gaussian_blur=args.gaussian_blur, gaussian_blur_axis=args.gaussian_blur_axis)


def main():
    qtim_commands()
