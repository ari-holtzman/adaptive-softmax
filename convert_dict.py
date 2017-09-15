import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dicpath', type=str, help='path to dictionary')
parser.add_argument('outpath', type=str, help='output path')
parser.add_argument('-m', default=2, help='minimum occurrences')
args = parser.parse_args()

freqs = {}
n_words = 0

with open(args.dicpath) as dic_file:
    for line in dic_file:
        index, word, num = line.split()
        if int(num) > args.m:
            freqs[word] = num
        else:
            freq['<unk>'] += num
        n_words += num



with open(args.outpath, 'w') as out_file:
    out_file.write('%d\n' % n_words)
    for word, count in sorted([ (k, v) for k, v in freqs.items() ]):
        out_file.write('%s %d\n' % (word, count))
