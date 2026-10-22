"""Compare a newly prepared run with the immutable archived split membership."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');p.add_argument('run',type=Path);a=p.parse_args()
    entries=json.loads((ROOT/'configs/experiments.json').read_text(encoding='utf-8'))
    e=next(e for e in entries if e['id']==a.experiment)
    differences=[]
    for fold in range(5):
        for part in ['train','test']:
            def ids(base):
                rows=json.loads((base/f'fold_{fold}'/f'{part}.json').read_text(encoding='utf-8-sig'))
                return [str(r['id']) for r in rows]
            old,new=ids(ROOT/e['results']),ids(a.run)
            if old!=new: differences.append(dict(fold=fold,partition=part,same_membership=set(old)==set(new),same_order=False))
    print(json.dumps({'differences':differences},indent=2))
    if differences: raise SystemExit(1)
if __name__=='__main__': main()
