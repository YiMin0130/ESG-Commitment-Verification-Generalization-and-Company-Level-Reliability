"""Rebuild supplement tables using only the Python standard library. Does not train models."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, csv, hashlib, json, math, statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'supplement'
FIELDS=['promise_status','verification_timeline','evidence_status','evidence_quality']
def read_json(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_csv(name,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    entries=read_json(ROOT/'configs/experiments.json')
    data=read_json(ROOT/'data/vpesg4k_train_val.json')
    canonical={str(r['id']):r for r in data}
    company=read_csv(ROOT/'data/company_ECR_CER_BHCR.csv')
    industries={r['ticker']:r['產業'] for r in company}
    assert len(canonical)==len(data)==2000
    splits=[]; labels=[]; membership=[]; fold_metrics=[]; inventory=[]; fingerprints={}; issues=[]
    for e in entries:
        d=ROOT/e['results']; seen=[]; signatures=[]
        for i in range(5):
            f=d/f'fold_{i}'; train=read_json(f/'train.json'); test=read_json(f/'test.json')
            ids=[set(str(r['id']) for r in rows) for rows in (train,test)]
            assert len(ids[0])==len(train) and len(ids[1])==len(test), (e['id'],i,'duplicate ID')
            assert not ids[0]&ids[1] and ids[0]|ids[1]==canonical.keys(),(e['id'],i,'ID partition')
            for rows in (train,test):
                for r in rows:
                    for k,v in canonical[str(r['id'])].items():
                        assert r[k]==v,(e['id'],i,r['id'],k)
            info=read_json(f/'split_info.json')
            for part,rows in [('train',train),('test',test)]:
                if part+'_indices' in info:
                    assert [data[j]['id'] for j in info[part+'_indices']]==[r['id'] for r in rows]
                for r in rows:
                    membership.append(dict(experiment=e['id'],fold=i,partition=part,id=r['id'],ticker=r['ticker'],industry=industries[str(r['ticker'])],report_sha256=hashlib.sha256(r['pdf_url'].encode()).hexdigest()))
                for field in FIELDS:
                    for label,n in sorted(Counter(r[field] for r in rows).items()):
                        labels.append(dict(experiment=e['id'],fold=i,partition=part,field=field,label=label,count=n))
            row=dict(experiment=e['id'],fold=i,n_train=len(train),n_test=len(test))
            for key,fn in [('company',lambda r:str(r['ticker'])),('industry',lambda r:industries[str(r['ticker'])]),('report',lambda r:r['pdf_url']),('text',lambda r:r['data'])]:
                a,b=({fn(r) for r in rows} for rows in (train,test))
                row[f'train_{key}']=len(a); row[f'test_{key}']=len(b); row[f'overlap_{key}']=len(a&b)
            row['overlap_id']=len(ids[0]&ids[1])
            if e['split']!='random': assert row['overlap_company']==row['overlap_report']==0
            if e['split']=='industry': assert row['overlap_industry']==0
            splits.append(row); seen.extend(str(r['id']) for r in test)
            signature=hashlib.sha256('\n'.join(sorted(ids[1])).encode()).hexdigest()
            signatures.append(signature)
        assert Counter(seen)==Counter(canonical.keys()),(e['id'],'OOF coverage')
        fingerprints[e['id']]=signatures
        for filename,group,metrics in [
            ('fold_classification_results.csv','field',['accuracy','balanced_accuracy','macro_f1']),
            ('fold_company_indicator_results.csv','indicator',['mae','rmse','spearman_rho','kendall_tau_b']),
            ('fold_regression_results.csv','indicator',['mae','rmse','spearman_rho','kendall_tau_b']),
            ('fold_pairwise_ranking_results.csv','indicator',['pairwise_accuracy','spearman_rho','kendall_tau_b'])]:
            p=d/filename
            if not p.exists(): continue
            for r in read_csv(p):
                for metric in metrics:
                    if metric not in r: continue
                    value=float(r[metric])
                    fold_metrics.append(dict(experiment=e['id'],split=e['split'],method=e['method'],fold=r['fold'],scope='text' if group=='field' else 'company',target=r[group],metric=metric,value=value,source=p.relative_to(ROOT).as_posix(),n=r.get('n_samples',r.get('n_companies',''))))
    stats=[]; grouped=defaultdict(list)
    for r in fold_metrics:
        grouped[(r['experiment'],r['split'],r['method'],r['scope'],r['target'],r['metric'])].append(r['value'])
    for key,values in sorted(grouped.items()):
        finite=[v for v in values if math.isfinite(v)]
        # Five-fold t critical value. Missing folds must not silently change degrees of freedom.
        n=len(finite); mean=statistics.mean(finite) if n else float('nan')
        sd=statistics.stdev(finite) if n>1 else float('nan')
        half=2.7764451051977987*sd/math.sqrt(5) if n==5 else float('nan')
        r=dict(zip(['experiment','split','method','scope','target','metric'],key))
        r.update(n_folds=n,mean=mean,sample_sd=sd,variance=sd*sd,descriptive_t95_low=mean-half,descriptive_t95_high=mean+half)
        stats.append(r)
        if n!=5: issues.append(f'{key}: {n} finite folds; interval unavailable')
    write_csv('fold_metrics.csv',fold_metrics); write_csv('mean_sd_descriptive_intervals.csv',stats)
    lookup={(r['experiment'],r['target'],r['metric']):r['mean'] for r in stats if r['scope']=='text'}
    gaps=[]
    for target in FIELDS:
        seen_mean=lookup[('random_qlora',target,'macro_f1')]
        for split in ['company','industry']:
            original=lookup[(split+'_qlora',target,'macro_f1')]
            contrastive=lookup[(split+'_contrastive',target,'macro_f1')]
            gaps.append(dict(target=target,split=split,seen_original_macro_f1=seen_mean,unseen_original_macro_f1=original,unseen_contrastive_macro_f1=contrastive,original_degradation=seen_mean-original,contrastive_degradation=seen_mean-contrastive,contrastive_minus_original=contrastive-original))
    write_csv('table2_generalization_gaps.csv',gaps)
    write_csv('split_overlap.csv',splits); write_csv('label_distributions.csv',labels); write_csv('split_membership.csv',membership)
    for split in ('random','company','industry'):
        q=fingerprints[split+'_qlora']
        for method in ('contrastive','ridge','pairwise'):
            same=q==fingerprints[split+'_'+method]
            if not same: issues.append(f'{split}_{method} test memberships differ from {split}_qlora; do not assume paired folds')
    (OUT/'split_fingerprints.json').write_text(json.dumps(fingerprints,indent=2)+'\n',encoding='utf-8')
    company_rows=[]
    for ticker,n in sorted(Counter(str(r['ticker']) for r in data).items()):
        rows=[r for r in data if str(r['ticker'])==ticker]
        company_rows.append(dict(ticker=ticker,company=rows[0]['company'],industry=industries[ticker],n_records=n,n_reports=len(set(r['pdf_url'] for r in rows))))
    write_csv('company_distribution.csv',company_rows)
    write_csv('industry_distribution.csv',[dict(industry=k,n_companies=sum(r['industry']==k for r in company_rows),n_records=sum(r['n_records'] for r in company_rows if r['industry']==k)) for k in sorted(set(industries.values()))])
    # A readable supplement with every task, method and split.
    lines=['# Detailed experimental results','', 'Generated from archived fold CSV files; no model was rerun.','', 'Intervals are mean ± t(0.975,4) × sample SD / sqrt(5), using unweighted folds. They describe fold variation, not independent-sample population confidence bounds. SD uses ddof=1; intervals are not clipped. OOF metrics in the archive are a different estimand.','', '| Experiment | Target | Metric | Mean | Sample SD | Descriptive 95% interval |','|---|---|---|---:|---:|---|']
    for r in stats:
        lines.append(f"| {r['experiment']} | {r['target']} | {r['metric']} | {r['mean']:.6f} | {r['sample_sd']:.6f} | [{r['descriptive_t95_low']:.6f}, {r['descriptive_t95_high']:.6f}] |")
    lines+=['','## Audit findings','']+[f'- {x}' for x in issues or ['All 60 train/test partitions passed ID coverage, content consistency and intended company/industry separation checks.']]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for p in sorted((ROOT/'src').glob('*.py')): ast.parse(p.read_text(encoding='utf-8'))
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='file_manifest.csv':
            inventory.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write_csv('file_manifest.csv',inventory)
    report=dict(experiments=len(entries),folds=len(splits),records=len(data),companies=len(company_rows),industries=len(set(industries.values())),metrics=len(fold_metrics),summaries=len(stats),issues=issues)
    (OUT/'audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
if __name__=='__main__': main()
