"""Independently check archived classification CSVs against saved predictions (stdlib only)."""
import csv,json,math
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABELS={'promise_status':['Yes','No'],'verification_timeline':['already','within_2_years','between_2_and_5_years','more_than_5_years','N/A'],'evidence_status':['Yes','No','N/A'],'evidence_quality':['Clear','Not Clear','Misleading','N/A']}
def main():
    checked=0; failures=[]
    for e in json.loads((ROOT/'configs/experiments.json').read_text(encoding='utf-8')):
        d=ROOT/e['results']; p=d/'fold_classification_results.csv'
        if not p.exists(): continue
        with p.open(encoding='utf-8-sig') as f: metrics=list(csv.DictReader(f))
        for i in range(5):
            preds=json.loads((d/f'fold_{i}/predictions.json').read_text(encoding='utf-8-sig'))
            gold=json.loads((d/f'fold_{i}/test.json').read_text(encoding='utf-8-sig'))
            by_id={str(r['id']):r for r in gold}
            assert Counter(str(r['id']) for r in preds)==Counter(by_id.keys()), (e['id'],i,'prediction membership')
            for field,labels in LABELS.items():
                pairs=[(r['gold_'+field],r['pred_'+field]) for r in preds]
                for r in preds: assert r['gold_'+field]==by_id[str(r['id'])][field]
                tp=Counter(a for a,b in pairs if a==b); actual=Counter(a for a,b in pairs); predicted=Counter(b for a,b in pairs)
                values={'accuracy':sum(tp.values())/len(pairs),'balanced_accuracy':sum(tp[k]/n for k,n in actual.items())/len(actual),'macro_f1':sum(2*tp[k]/(actual[k]+predicted[k]) if actual[k]+predicted[k] else 0 for k in labels)/len(labels)}
                archived=next(r for r in metrics if int(r['fold'])==i and r['field']==field)
                for metric,v in values.items():
                    checked+=1
                    if not math.isclose(v,float(archived[metric]),abs_tol=1e-12): failures.append(dict(experiment=e['id'],fold=i,field=field,metric=metric,computed=v,archived=archived[metric]))
    report=dict(checked=checked,failures=failures)
    (ROOT/'supplement/prediction_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if failures: raise SystemExit(1)
if __name__=='__main__': main()
