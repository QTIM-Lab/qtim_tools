import argparse
import sys

import qtim_tools

class qtim_commands(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='A number of pre-packaged command used by the Quantiative Tumor Imaging Lab at the Martinos Center',
            usage='''git <command> [<args>]

The following commands are available:
   commit     Record changes to the repository
   fetch      Download objects and refs from another repository
''')

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print 'Sorry, that\'s not one of the commands.'
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def label_statistics(self):
        parser = argparse.ArgumentParser(
            description='Generate label statistics from a QTIM study and a given ROI.')

        parser.add_argument('-study_name')
        parser.add_argument('-label_name')
        parser.add_argument('-output_csv')

        args = parser.parse_args(sys.argv[2:])
        print 'Running label statistics on study directory... %s' % args.study_name

        qtim_tools.qtim_pipelines.label_statistics.qtim_study_statistics(args.study_name, args.label_name, args.output_csv)

    # def fetch(self):
    #     parser = argparse.ArgumentParser(
    #         description='Download objects and refs from another repository')
    #     # NOT prefixing the argument with -- means it's not optional
    #     parser.add_argument('repository')
    #     args = parser.parse_args(sys.argv[2:])
    #     print 'Running git fetch, repository=%s' % args.repository

def main():
    qtim_commands()