
with open('BoundedCLIP.txt', 'r') as bounded_f:
    idx_check = {}
    for idx,line in enumerate(bounded_f):
        if idx == 0:
            continue
        line = line.strip().split(',')
        idx_check[line[0]] = {
            'wrong': line[1],
            'correct': line[2]
        }

with open('ActionCLIP_conf.txt', 'r') as action_f:
    indices = []
    with open('compare.txt', 'w') as f:
        f.write('idx,bounded,action,correct\n')
        for idx, line in enumerate(action_f):
            if idx==0:
                continue
            line = line.strip().split(',')
            if line[0] in list(idx_check.keys()):
                bounded_misclass = idx_check[line[0]]['wrong']
                if bounded_misclass != line[1]:
                    f.write(f'{line[0]},{bounded_misclass},{line[1]},{line[2]}\n')
            elif line[0] not in idx_check:
                f.write(f'{line[0]},,{line[1]},{line[2]}\n')
            indices.append(line[0])
    
        for key in idx_check.keys():
            if key not in indices:
                idx=key
                corr = idx_check[key]['correct']
                wrong = idx_check[key]['wrong']
                f.write(f'{idx},{wrong},,{corr}\n')