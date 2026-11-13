import ast, hashlib, json
from pathlib import Path

v2=Path('../fl_dagsnet_veremi_v2')
class StripDocs(ast.NodeTransformer):
 def visit_Module(self,node): return self.body(node)
 def visit_ClassDef(self,node): return self.body(node)
 def visit_FunctionDef(self,node): return self.body(node)
 def body(self,node):
  self.generic_visit(node)
  if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
  return node
 def visit_Assert(self,node):
  self.generic_visit(node);node.msg=None;return node

def normalized(s):return ast.dump(StripDocs().visit(ast.parse(s)),include_attributes=False)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={}
for name in ['core/data.py','core/fl.py','core/optim.py','core/metrics.py','core/evaluate.py']:
 a,b=v2/name,Path(name)
 assert normalized(a.read_text())==normalized(b.read_text()),name
 report[name]={'check':'same executable AST','v2_sha256':digest(a),'bert_sha256':digest(b)}
for b in sorted(Path('methods').glob('*.py')):
 if b.name in {'apfed.py','v2x_htfl.py','pfedes.py','lwfednids.py'}:continue
 a=v2/b
 assert normalized(a.read_text())==normalized(b.read_text()),str(b)
 report[str(b)]={'check':'same executable AST','v2_sha256':digest(a),'bert_sha256':digest(b)}
a=(v2/'methods/apfed.py').read_text().replace('"head.5"','"head."')
b=Path('methods/apfed.py')
assert normalized(a)==normalized(b.read_text())
report[str(b)]={'check':'same AST after classifier path head.5 -> head.','bert_sha256':digest(b)}
a=(v2/'methods/v2x_htfl.py').read_text().replace('core.dagsnet','core.bert').replace('DAGSNet','FeatureBERT')
b=Path('methods/v2x_htfl.py')
assert normalized(a)==normalized(b.read_text())
report[str(b)]={'check':'same AST after base model import/class substitution','bert_sha256':digest(b)}
a=ast.parse((v2/'methods/pfedes.py').read_text());b=ast.parse(Path('methods/pfedes.py').read_text())
a=next(n for n in a.body if isinstance(n,ast.ClassDef) and n.name=='Method')
b=next(n for n in b.body if isinstance(n,ast.ClassDef) and n.name=='Method')
assert ast.dump(StripDocs().visit(a))==ast.dump(StripDocs().visit(b))
report['methods/pfedes.py']={'check':'Method AST identical; 7x11 proxy architecture and grouped Conv1d verified by tests/test_units.py'}
a=(v2/'scripts/run_fl.py').read_text().replace('from core.dagsnet import width','from core.bert import cfg_full')
a=a.replace('    ap.add_argument("--width", type=float, default=float(os.environ.get("DAGS_WIDTH", 1.0)))\n','')
a=a.replace('cfg = width(a.width)','cfg = cfg_full()')
assert normalized(a)==normalized(Path('scripts/run_fl.py').read_text())
report['scripts/run_fl.py']={'check':'same AST after model factory and DAGS_WIDTH argument substitution'}
Path('docs/research/v2-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print('V2_PARITY_PASS',len(report),'checks; LwFedNIDS/model engine are architecture-specific and separately tested')
