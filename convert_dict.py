import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dicpath', type=str, help='path to dictionary')
parser.add_argument('outpath', type=str, help='output path')
parser.add_argument('-m', default=10, help='minimum occurrences')
args = parser.parse_args()

freqs = {}
n_words = 0

with open(args.dicpath) as dic_file:
    for line in dic_file:
        index, word, num = line.split(' ')
        num = int(num)
        if len(freqs) < 2 or num > args.m:
            freqs[word] = num
        else:
            freqs['<unk>'] += num
        n_words += num



with open(args.outpath, 'w') as out_file:
    out_file.write('%d\n' % n_words)
    for count, word in sorted([ (v, k) for k, v in freqs.items() ])[::-1]:
        out_file.write('%s %d\n' % (word, count))
