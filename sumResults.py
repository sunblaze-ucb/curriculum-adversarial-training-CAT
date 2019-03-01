from __future__ import division
import sys, os
import numpy as np

result_dirs = sys.argv[1:]

# sum results
results_folders = []
def sumResults():
    for md in result_dirs:
        if md[-1]=='/': md = md[:-1]
        print('md', md)
        if md[-7:]=='results':
            results_folders.append(md)

    print('results_folders', results_folders)
    # gen sum result file first
    for fd in results_folders:
        print('fd', fd)
        results_sum = {'test.quant0':[], 'test.quant1':[], 'train.quant0':[], 'train.quant1':[], 'valid.quant0':[], 'valid.quant1':[]}
        loss_sum = dict((k+'.loss', []) for k in results_sum.keys() )
        this_dir = fd
        rstfns = [fn for fn in os.listdir(this_dir)]
        for fn in rstfns:
            tmp = fn.split('.')
            if len(tmp)==3 and tmp[-1] != 'loss':  
                print('fn', fn)
                results_sum[tmp[0]+'.'+tmp[-1]].append(fn)
            elif len(tmp)==4: loss_sum[tmp[0]+'.'+tmp[-2]+'.'+tmp[-1]].append(fn)

        #print loss_sum
        for sumf, subfs in loss_sum.items():
            if len(subfs)==0: continue
            subfs = sorted(subfs, key = lambda x : int(x.split('.')[1].split('_')[0]))
            #print(sumf, subfs)
            loss_arr = None
            for sf in subfs:
                this_fn = os.path.join(this_dir, sf)
                arr = np.genfromtxt(this_fn, delimiter='\t')[1:].astype('float') # first ln is col name
                if loss_arr is None:
                    loss_arr = arr
                    with open(this_fn, 'r') as fp: firstln = fp.readlines()[0]
                else:
                    loss_arr = np.concatenate((loss_arr, arr), axis=0)
            with open(os.path.join(this_dir, sumf), 'w') as fp_sum:
                fp_sum.write(firstln)
                np.savetxt(fp_sum, loss_arr, fmt='%.2f', delimiter='\t') 

        print results_sum
        for sumf, subfs in results_sum.items():
            if len(subfs)==0: continue
            subfs = sorted(subfs, key = lambda x : int(x.split('.')[1].split('_')[0]))
            print(sumf, subfs)
            sum_arr, sum_loss, firstln = None, None, None
            for sf in subfs:
                this_fn = os.path.join(this_dir, sf)
                arr = np.genfromtxt(this_fn, delimiter='\t')[1:] # first ln is col name
                if sum_arr is None:
                    sum_arr = arr[2:]
                    sum_loss = arr[1]
                    with open(this_fn, 'r') as fp: firstln = fp.readlines()[0]
                else:
                    sum_arr = np.concatenate((sum_arr, arr[2:]), axis=0)
                    sum_loss += arr[1]
                 
            print('result file to write', os.path.join(this_dir, sumf))               
            with open(os.path.join(this_dir, sumf), 'w') as fp_sum:
                fp_sum.write(firstln)
                avg_acc = sum_arr.sum(axis=0)/len(sum_arr)
                np.savetxt(fp_sum, np.concatenate((np.expand_dims(avg_acc*100, axis=0), np.expand_dims(sum_loss, axis=0), sum_arr), axis=0), fmt='%.f', delimiter='\t') 
                     

sumResults()
