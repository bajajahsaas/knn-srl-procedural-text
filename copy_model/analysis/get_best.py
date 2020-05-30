import pickle
import numpy as np
import sys


relations = open('../relations.txt','r').read().splitlines() + ['No-Relation']

def print_example(w):
    print(w['sentence'])
    print('Neighbors:')
    print('\n'.join(w['context_sents']))
    qh,qt = w['edges']
    max_attn = np.argmax(w['attention'], axis=2)[0,:]
    ch,ct = w['context_edges']
    best_ch = [ch[i] if i < len(ch) else 'Prototype' for i in max_attn]
    best_ct = [ct[i] if i < len(ch) else 'Prototype' for i in max_attn]
    best_cl = [w['context_labels'][0][i] if i <len(ch) else 'Prototype' for i in max_attn]
    print('Queries:')
    for qh1, qt1, ch1,ct1,cl1, gt, pred in zip(qh,qt,best_ch,best_ct,best_cl\
                                           ,w['ground_truth'][0], w['pred']):
        if cl1 == 'Prototype':
            context_rel = 'Prototype'
        else:
            context_rel = relations[cl1]
        print('query:\t\t %s->%s\nbest_context:\t%s->%s\nbest_context_label:\t%s\nground_truth:\t%s\nprediction:\t%s\n'\
              %(qh1,qt1,ch1,ct1, context_rel,relations[gt],relations[pred]))

fil = sys.argv[1]
with open(fil, 'rb') as f:
    data = pickle.load(f)

sorted_data = sorted(data,key=lambda x:x['sent_f1'])
worst = sorted_data[:100]
print('WORST:')
for w in worst:
    print_example(w)
    print('\n\n\n\n\n\n')


print('BEST:')
best = sorted_data[-100:][::-1]
for w in best:
    print_example(w)
    print('\n\n\n\n\n\n')


