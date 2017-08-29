import argparse, os
from subprocess import check_output

parser = argparse.ArgumentParser()
parser.add_argument('text_dir', type=str, help='directory with text files')
parser.add_argument('model_dir', type=str, help='directory with model files')
args = parser.parse_args() 

model_dir_path = os.fsencode(args.model_dir)
text_dir_path = os.fsencode(args.text_dir)
filenames = next(os.walk(text_dir_path))[2]
for filename in filenames:
    name, extension = os.path.splitext(filename)
    name, extension = name.decode('utf-8'), extension.decode('utf-8')
    if extension == '.txt':
        text_path = os.path.join(text_dir_path, filename)
        cur_args = ['th', 'get_probs.lua',
                '-modeldir', args.model_dir,
                '-textpath', text_path]
        output = check_output(cur_args)
        out_filename = '%s_probs%s' % (name, extension)
        out_path = os.path.join(args.text_dir, out_filename)
        with open(out_path, 'wb') as out_file:
            out_file.write(output)
