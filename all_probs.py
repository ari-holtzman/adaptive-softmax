import argparse, os
from subprocess import run

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
        args = ['th', 'get_probs.lua',
                '-modeldir', args.model_dir,
                '-textpath', text_path]
        comp_proc = run(args)
        out_filename = '%s_probs%s' % (name, extension)
        with open(out_filename, 'wb') as out_file:
            out_file.write(comp_proc.stdout)
