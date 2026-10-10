#!/usr/bin/env python3
import json,os,subprocess,sys,time
from pathlib import Path
r=Path.cwd(); state=r/'.runtime/ralph-supervisor.json'; state.parent.mkdir(exist_ok=True)
prd=json.loads((r/'prd.json').read_text()); pending=sorted((s for s in prd['userStories'] if not s['passes']),key=lambda s:s['priority'])
if not pending: sys.exit(0)
sid=pending[0]['id']; old=json.loads(state.read_text()) if state.exists() else {}
if sid == 'US-009':
 review_path = r/'.runtime/batch3-supervisor-review.json'
 head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 try:
  review = json.loads(review_path.read_text())
 except (OSError,ValueError):
  review = {}
 if review.get('implementation_commit') != head or review.get('US-007') is not True or review.get('US-008') is not True:
  print('STOP: US-009 awaits supervisor review of US-007 and US-008 at the current commit; no model request started')
  sys.exit(6)
if old.get('story')==sid and old.get('failures',0)>=2:
 print('STOP: two failed attempts require supervisor review',sid);sys.exit(2)
dirty=subprocess.check_output(['git','status','--porcelain'],text=True)
if dirty:
 print('STOP: dirty worktree requires supervisor review');sys.exit(3)
prompt=Path(sys.argv[1]).read_text()+f"\nThis iteration is restricted to {sid}. No other story. Run focused verification and commit evidence locally."
if sid == 'US-009':
 prompt += "\nUS-009 PREPARATION PHASE ONLY: implement the complete real-acceptance runner/config and meaningful OFFLINE tests, commit locally, leave US-009 passes=false, and STOP with exact invocation/evidence. Do NOT create/start the persistent acceptance ledger and do NOT send any real model-generation request. The supervisor will review the committed runner, then execute the one real acceptance itself. Coding CLI usage is separate. Live mode must refuse before ledger/auth/model use unless an ignored supervisor review file binds the current implementation commit; offline/preflight mode must not require or create a live ledger. Do not manufacture that review file. Preparation ending with US009 false is expected, not a failed story."
try: result=subprocess.run([str(Path.home() / '.kimi-code/bin/kimi'),'-p',prompt],timeout=(7200 if sid in {'US-008', 'US-009'} else 3600))
except subprocess.TimeoutExpired:
 failures=(old.get('failures',0) if old.get('story')==sid else 0)+1
 state.write_text(json.dumps(dict(story=sid,failures=failures,updated=time.time(),reason='programming_timeout')))
 print('STOP: programming iteration timeout; recorded failed attempt',sid);sys.exit(4)
now=json.loads((r/'prd.json').read_text()); passed=next(s for s in now['userStories'] if s['id']==sid)['passes']
if sid == 'US-009' and not passed and result.returncode == 0:
 state.write_text(json.dumps(dict(story=sid,failures=0,updated=time.time(),reason='awaiting_runner_supervisor_review')))
 print('US-009 preparation finished; supervisor must review and execute real acceptance; story remains incomplete')
 sys.exit(0)
failures=0 if passed else (old.get('failures',0) if old.get('story')==sid else 0)+1
state.write_text(json.dumps(dict(story=sid,failures=failures,updated=time.time())))
if not passed:print('STOP: story remains failing',sid);sys.exit(5)
if result.returncode:sys.exit(result.returncode)
