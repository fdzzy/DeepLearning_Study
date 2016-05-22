#!/usr/bin/python

'''
http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

Unsmoothed Maximum Likelihood Character Level Language Model
'''

from collections import *
from random import random

#INPUT_FILE = r'data/shakespeare_input.txt'
#INPUT_FILE = r'D:\source\Linux\word2vec\trunk\text8'
INPUT_FILE = r'data/linux_input.txt'
LM_ORDER = 10

def train_char_lm(fname, order=4):
    data = file(fname).read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in xrange(len(data) - order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char] += 1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c, cnt/s) for c, cnt in counter.iteritems()]
    outlm = {hist:normalize(chars) for hist, chars in lm.iteritems()}
    return outlm
    
def generate_letter(lm, history, order):
    history = history[-order:]
    dist = lm[history]
    x = random()
    for c, v in dist:
        x = x - v
        if x <= 0: return c
        
def generate_text(lm, order, nletters=1000):
    history = "~" * order
    out = []
    for i in xrange(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)

    
def main():
    lm = train_char_lm(INPUT_FILE, order=LM_ORDER)
    #print lm['ello']
    #print lm['Firs']
    #print lm['rst ']
    print generate_text(lm, LM_ORDER)
    
main()