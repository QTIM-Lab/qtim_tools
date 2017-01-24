import os
import glob

if __name__ == '__main__':
	for file in glob.glob('C:/Users/abeers/Documents/Projects/Texture_Analysis_Class/Masks/*'):
		split_file = str.split(file, '.')
		newfile = split_file[0] + '-label.' + split_file[1] + '.' + split_file[2]
		print newfile
		os.rename(file, newfile)
