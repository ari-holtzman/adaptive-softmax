import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dicpath', type=str, 'path to dictionary')
parser.add_argument('outpath', type=str, 'output path')
parser.add_argument('-m', default=2, 'minimum occurrences')
args = parser.parse_args()

freqs = {}
n_words = 0

with open(dicpath) as dic_file:
    for line in dic_file:
        index, word, num = word.split()
        if num > args.m then
            freqs[word] = num
        else:
            freq['<unk>'] += num
        n_words += num



with open(outpath, 'w') as out_file:
    out_file.write('%d\n' % n_words)
    for word, count in sorted([ k, v for freqs.items() ]):
        out_file.write('%s %d\n' % (word, count))
