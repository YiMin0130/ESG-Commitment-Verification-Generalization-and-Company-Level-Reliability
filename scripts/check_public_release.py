"""Scan release files for private machine paths, common credential formats and unexpected binaries."""
from pathlib import Path
import ast,json,re,struct,zipfile
ROOT=Path(__file__).resolve().parents[1]
RULES={
 'private_user_path':re.compile(r'(?<![A-Za-z0-9])/(?:home|Users|root)/[^\s\x22\x27`]+'),
 'windows_absolute_path':re.compile(r'\b[A-Za-z]:[\\/][^\s\x22\x27`]+'),
 'unc_path':re.compile(r'\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_$.-]+'),
 'private_ip':re.compile(r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b'),
 'credential':re.compile(r'(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')}
URL=re.compile(r'https?://[^\s\x22\x27<>`]+')
def main():
 findings=[];scanned=0;caches=[]
 for p in sorted(ROOT.rglob('*')):
  if p.is_symlink():findings.append(dict(path=p.relative_to(ROOT).as_posix(),rule='symbolic_link'));continue
  if not p.is_file():continue
  rel=p.relative_to(ROOT).as_posix();scanned+=1
  if any(part in {'.git','.aws','.ssh','__pycache__','.venv'} for part in p.relative_to(ROOT).parts) or p.name=='.env':
   findings.append(dict(path=rel,rule='private_or_generated_directory'))
  if p.suffix=='.npz':
   members=[]
   with zipfile.ZipFile(p) as z:
    if z.comment:findings.append(dict(path=rel,rule='npz_archive_comment'))
    for item in z.infolist():
     with z.open(item) as f:
      magic=f.read(8)
      assert magic[:6]==b'\x93NUMPY','Invalid array format'
      n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0]
      header=ast.literal_eval(f.read(n).decode('latin1').strip())
      if item.filename not in ['ids.npy','embeddings.npy'] or 'O' in str(header['descr']):findings.append(dict(path=rel,rule='unexpected_or_object_array'))
      members.append(dict(name=item.filename,dtype=header['descr'],shape=header['shape']))
   caches.append(dict(path=rel,arrays=members));continue
  try:text=p.read_text(encoding='utf-8-sig')
  except UnicodeError:findings.append(dict(path=rel,rule='unexpected_binary'));continue
  text=URL.sub('[PUBLIC_URL]',text)
  for rule,pattern in RULES.items():
   if pattern.search(text):findings.append(dict(path=rel,rule=rule))
 report=dict(files_scanned=scanned,findings=findings,embedding_caches=caches,scope='Static paths, credential patterns, binary array headers and unexpected file types. Public source text/URLs and team/paper attribution are retained; this does not certify author anonymity or dataset redistribution rights.')
 (ROOT/'supplement/privacy_scan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(files_scanned=scanned,findings=len(findings),embedding_caches=len(caches)),indent=2))
 if findings:
  print(json.dumps(findings,indent=2));raise SystemExit(1)
if __name__=='__main__':main()
