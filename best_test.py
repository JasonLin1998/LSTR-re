import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Find best loss")
    parser.add_argument("--exp", help="the name of exp", type=str, default="try")
    parser.add_argument("--mode", help="train or val", type=str, default="train")
    args = parser.parse_args()
    return args


def find_best(exp, mode, find_pkl):
    best = 100
    results = {}
    if not find_pkl:
        with open('./experiment/{}/save{}_loss.log'.format(exp, mode), 'r') as f:
            for ind, line in enumerate(f.readlines()):
                line.strip()
                lines = line.split()
                if not ind == 0:
                    if float(lines[1]) <= float(best):
                        best = float(lines[1])
                        iter = int(lines[0])

        # print('exp: {}  mode: {}  iter: {}  loss: {}'.format(exp, mode, iter, best))
        results['exp'] = exp
        results['mode'] = mode
        results['iter'] = iter
        results['loss'] = best

    else:
        with open('./experiment/{}/save{}_loss.log'.format(exp, mode), 'r') as f:
            for ind, line in enumerate(f.readlines()):
                line.strip()
                lines = line.split()
                if not ind == 0:
                    if int(lines[0]) % 5000 == 0:
                        if float(lines[1]) <= float(best):
                            best = float(lines[1])
                            iter = int(lines[0])
        results['exp'] = exp
        results['mode'] = mode
        results['iter'] = iter
        results['loss'] = best

    return results

def write_info(name, mode, find_pkl):
    for i in range(0, 6):
        if i == 1:
            results = find_best(name[i], mode, find_pkl)
            with open('./best_results.txt', 'a') as f:
                f.write('{}\t\t{}\t\t{}\t\t{}\n'.format(results['exp'], results['mode'], results['iter'], results['loss']))

if __name__ == '__main__':
    args = parse_args()
    name = []
    exp0 = 'origin'
    exp1 = 'try10ss'
    exp2 = 'try12'
    exp3 = 'try13'
    exp4 = 'try40'
    exp5 = 'try50'
    mode1 = 'train'
    mode2 = 'val'
    find_pkl = True

    name.append(exp0)
    name.append(exp1)
    name.append(exp2)
    name.append(exp3)
    name.append(exp4)
    name.append(exp5)

    with open('./best_results.txt', 'w') as a:
        a.write('exp\t\tmode\t\titer\t\tloss\n')

    write_info(name, mode1, find_pkl)
    write_info(name, mode2, find_pkl)

