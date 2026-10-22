"""Portable launcher. Dry-run is dependency-free; outputs never default to archived results."""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', nargs='?')
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--gpu', default='0')
    p.add_argument('--data', type=Path, default=ROOT/'data/vpesg4k_train_val.json')
    p.add_argument('--output-dir', type=Path)
    a, extra=p.parse_known_args()
    entries=json.loads((ROOT/'configs/experiments.json').read_text(encoding='utf-8'))
    by_id={e['id']:e for e in entries}
    if a.list:
        print('\n'.join(sorted(by_id))); return
    if a.experiment not in by_id: p.error('Choose an experiment with --list.')
    e=by_id[a.experiment]
    out=(a.output_dir or ROOT/'runs'/a.experiment).resolve()
    if out == (ROOT/'results').resolve() or (ROOT/'results').resolve() in out.parents:
        p.error('Archived results are read-only inputs. Choose an output outside results/.')
    if out.exists() and any(out.iterdir()): p.error('Output is not empty; choose a new output directory.')
    data=a.data.resolve()
    if not data.is_file(): p.error(f'Missing data: {data}')
    cmd=[sys.executable,str(ROOT/e['script']),'--data',str(data),'--output-dir',str(out),'--gpu',a.gpu]
    if '--company-info' in e['arguments']:
        cmd+=['--company-info',str(ROOT/'data/company_ECR_CER_BHCR.csv')]
    cmd+=extra
    # Display a portable command; the subprocess still uses resolved paths internally.
    def display_path(value):
        path=Path(value)
        if path.is_absolute():
            try:return path.relative_to(ROOT).as_posix()
            except ValueError:return '<external-path>'
        return value
    public_cmd=['python']+[display_path(v) for v in cmd[1:]]
    print(subprocess.list2cmdline(public_cmd),flush=True)
    if not a.dry_run:
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=a.gpu
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__': main()
