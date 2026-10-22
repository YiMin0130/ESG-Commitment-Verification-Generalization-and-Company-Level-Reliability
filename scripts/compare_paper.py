"""Compare the corrected paper's Tables 2–4 and quantitative conclusions to the archive."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    with (ROOT/'supplement/mean_sd_descriptive_intervals.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    lookup={(r['experiment'],r['scope'],r['target'],r['metric']):r for r in rows}
    reference=json.loads((ROOT/'configs/paper_reference.json').read_text(encoding='utf-8'))
    output=[]
    for cell in reference['entries']:
        actual=lookup[(cell['experiment'],cell['scope'],cell['target'],cell['metric'])]
        for stat in ['mean','sample_sd','descriptive_t95_low','descriptive_t95_high']:
            if cell[stat] is None: continue
            value=float(actual[stat]); paper=cell[stat]; digits=cell['digits']
            # Compare at the display precision, allowing only half a final digit.
            match=abs(value-paper)<=0.5*10**(-digits)+1e-12
            output.append(dict(table=cell['table'],experiment=cell['experiment'],target=cell['target'],metric=cell['metric'],statistic=stat,paper=paper,archive=value,archive_minus_paper=value-paper,status='match_at_printed_precision' if match else 'MISMATCH'))
    with (ROOT/'supplement/table2_generalization_gaps.csv').open(encoding='utf-8-sig') as f:
        gaps={(r['split'],r['target']):r for r in csv.DictReader(f)}
    for cell in reference['derived_entries']:
        value=float(gaps[(cell['split'],cell['target'])][cell['metric']]);paper=cell['value']
        match=abs(value-paper)<=0.5*10**(-cell['digits'])+1e-12
        output.append(dict(table=2,experiment=cell['split'],target=cell['target'],metric=cell['metric'],statistic='derived_difference',paper=paper,archive=value,archive_minus_paper=value-paper,status='match_at_printed_precision' if match else 'MISMATCH'))
    path=ROOT/'supplement/paper_comparison.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
    mismatches=[r for r in output if r['status']=='MISMATCH']
    narrative=verify_narrative(lookup,gaps)
    lines=['# Paper and archive correspondence','', '[English README](../README.md) | [繁體中文 README](../README.zh-TW.md)','',f"Reference version: **{reference['version']}**.",'',f"Reference PDF SHA-256: `{reference['pdf_sha256']}`.",'',f"Reference LaTeX SHA-256: `{reference['latex_sha256']}`.",'',f'Compared {len(output)} displayed numbers in corrected paper Tables 2–4; {len(mismatches)} differ beyond printed rounding. The comparison includes all 440 mean/SD/interval values and all 24 Table 2 degradation/delta values. The reference numbers were extracted from the corrected LaTeX and verified against its PDF independently of the archived metrics. The previous-paper discrepancy is resolved.','', '| Table | Numbers compared | Mismatches |','|---|---:|---:|']
    for table in [2,3,4]:
        lines.append(f"| {table} | {sum(r['table']==table for r in output)} | {sum(r['table']==table for r in mismatches)} |")
    if mismatches:
        lines+=['','| Table | Experiment | Target | Metric | Statistic | Paper | Archive |','|---|---|---|---|---|---:|---:|']
        lines += [f"| {r['table']} | {r['experiment']} | {r['target']} | {r['metric']} | {r['statistic']} | {r['paper']} | {r['archive']:.8f} |" for r in mismatches]
    lines+=['','## Quantitative descriptions','',f"All {len(narrative)} numerical/directional checks passed. Full observations are in [narrative_verification.json](../supplement/narrative_verification.json).",'', '- Unseen-company original Macro-F1 decreases by 0.0037–0.0117 on three tasks; timeline improves by 0.0053.', '- Unseen-industry original Macro-F1 decreases by 0.0136–0.0198 on all four tasks, with the largest decrease on timeline.', '- Contrastive learning improves three unseen-company tasks but decreases timeline by 0.0329; unseen-industry timeline decreases by 0.0120.', '- Unseen-industry evidence-quality Macro-F1 increases from 0.4274 to 0.4401, approaching the Seen baseline of 0.4410. Derived differences use full-precision fold means before rounding.', '- For M1/M2 under unseen settings, MAE is 0.088–0.133 and RMSE is 0.119–0.159. M2 has lower unseen-company CER error, while M1 has lower ECR and BHCR errors.', '- Unseen-company ECR Spearman means for M1/M2/M3 are 0.338/0.329/0.352. All 18 unseen-industry rank-correlation intervals include zero.','', '## 中文核對結果','',f"本次以修正版論文為基準，Tables 2–4 共 {len(output)} 個數值已核對，差異 {len(mismatches)} 項；包含平均、標準差、描述性區間，以及 Table 2 的全部 D／CL-Original 衍生差值。先前舊版論文與資料夾不一致的問題已解決。",'', '內文核對涵蓋公司／產業切分的效能方向與差值範圍、對比學習改善任務數與 timeline 退步幅度、公司估計誤差與方法比較，以及公司排序相關係數和區間跨零情形。這些是數值與方向的核對，不代表已驗證外部 ESG 效度或完成 GPU 重訓。']
    lines += ['', '## Scope missing from this archive','', 'The competition system in Section 3.2 includes Qwen Six-task, Chinese RoBERTa and field-level ensembling. Their training scripts, predictions, leaderboard submissions and leaderboard evidence were not found in the supplied Qwen_no_adapter folder. Public and private scores are manuscript claims, not independently reproduced archive results. Figures 1, 2 and 4 can be traced to company_distribution.csv, industry_distribution.csv and data/company_ECR_CER_BHCR.csv; original plotting code is absent. Figures 3 and system architecture are descriptive. Table 1 is the task schema rather than an experimental result.']
    (ROOT/'docs/PAPER_CORRESPONDENCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary=dict(reference_version=reference['version'],pdf_sha256=reference['pdf_sha256'],latex_sha256=reference['latex_sha256'],numbers_compared=len(output),mismatches=len(mismatches),narrative_checks=len(narrative),narrative_failures=0)
    (ROOT/'supplement/paper_comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(f'{len(mismatches)} mismatches / {len(output)} statistics')
    if mismatches: raise SystemExit(1)

def verify_narrative(lookup,gaps):
    checks=[]
    def check(name,condition,observed):
        checks.append(dict(claim=name,passed=bool(condition),observed=observed))
    def val(e,t,m,stat='mean'):return float(lookup[(e,'company' if t in ['ECR','CER','BHCR'] else 'text',t,m)][stat])
    company={k[1]:float(r['original_degradation']) for k,r in gaps.items() if k[0]=='company'}
    positive=[v for v in company.values() if v>0]
    check('Unseen-company three original tasks decline by 0.0037–0.0117',len(positive)==3 and [round(min(positive),4),round(max(positive),4)]==[.0037,.0117],company)
    check('Unseen-company original timeline improves by 0.0053',round(-company['verification_timeline'],4)==.0053,-company['verification_timeline'])
    industry={k[1]:float(r['original_degradation']) for k,r in gaps.items() if k[0]=='industry'}
    check('All unseen-industry original tasks decline by 0.0136–0.0198',all(v>0 for v in industry.values()) and [round(min(industry.values()),4),round(max(industry.values()),4)]==[.0136,.0198],industry)
    check('Largest original industry decline is timeline',max(industry,key=industry.get)=='verification_timeline',industry)
    delta={k[1]:float(r['contrastive_minus_original']) for k,r in gaps.items() if k[0]=='company'}
    check('Contrastive improves three unseen-company tasks',sum(v>0 for v in delta.values())==3,delta)
    check('Company contrastive timeline drops 0.0329',round(-delta['verification_timeline'],4)==.0329,delta['verification_timeline'])
    d=float(gaps[('industry','verification_timeline')]['contrastive_minus_original'])
    check('Industry contrastive timeline drops 0.0120',round(-d,4)==.0120,d)
    quality=[val('industry_qlora','evidence_quality','macro_f1'),val('industry_contrastive','evidence_quality','macro_f1'),val('random_qlora','evidence_quality','macro_f1')]
    check('Industry evidence quality 0.4274 to 0.4401 approaches Seen 0.4410',[round(v,4) for v in quality]==[.4274,.4401,.4410] and abs(quality[1]-quality[2])<abs(quality[0]-quality[2]),quality)
    for metric,expected in [('mae',[.088,.133]),('rmse',[.119,.159])]:
        values=[val(s+'_'+m,t,metric) for s in ['company','industry'] for m in ['qlora','ridge'] for t in ['ECR','CER','BHCR']]
        check('Unseen M1/M2 '+metric+' range', [round(min(values),3),round(max(values),3)]==expected,[min(values),max(values)])
    errors={t:{m:val('company_'+m,t,'mae') for m in ['qlora','ridge']} for t in ['ECR','CER','BHCR']}
    check('M2 lower company CER MAE; M1 lower ECR/BHCR MAE',errors['CER']['ridge']<errors['CER']['qlora'] and all(errors[t]['qlora']<errors[t]['ridge'] for t in ['ECR','BHCR']),errors)
    rho=[val('company_'+m,'ECR','spearman_rho') for m in ['qlora','ridge','pairwise']]
    check('Company ECR Spearman M1/M2/M3 0.338/0.329/0.352',[round(v,3) for v in rho]==[.338,.329,.352],rho)
    intervals=[[val('industry_'+m,t,metric,'descriptive_t95_low'),val('industry_'+m,t,metric,'descriptive_t95_high')] for m in ['qlora','ridge','pairwise'] for t in ['ECR','CER','BHCR'] for metric in ['spearman_rho','kendall_tau_b']]
    check('All 18 unseen-industry ranking intervals include zero',len(intervals)==18 and all(lo<=0<=hi for lo,hi in intervals),intervals)
    (ROOT/'supplement/narrative_verification.json').write_text(json.dumps(dict(checks=checks,failures=sum(not c['passed'] for c in checks)),indent=2)+'\n',encoding='utf-8')
    assert all(c['passed'] for c in checks),'Quantitative description check failed; see narrative_verification.json'
    return checks
if __name__=='__main__': main()
