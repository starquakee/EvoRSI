#!/usr/bin/env python3
import json,os,subprocess,sys,time
from pathlib import Path
r=Path.cwd(); state=r/'.runtime/comparison-coding-supervisor.json'; state.parent.mkdir(exist_ok=True)
prd=json.loads((r/'prd.json').read_text()); pending=sorted((s for s in prd['userStories'] if not s['passes']),key=lambda s:s['priority'])
if not pending: sys.exit(0)
sid=pending[0]['id']; old=json.loads(state.read_text()) if state.exists() else {}
if old.get('story')==sid and old.get('failures',0)>=2:
 print('STOP: two failed attempts require supervisor review',sid);sys.exit(2)
dirty=subprocess.check_output(['git','status','--porcelain'],text=True)
if dirty:
 print('STOP: dirty worktree requires supervisor review');sys.exit(3)
prompt=Path(sys.argv[1]).read_text()+f"\nThis iteration is restricted to {sid}. No other story. Run focused verification and commit evidence locally."
if sid == 'EXP-008':
 prompt += "\nEXP-008 PREPARATION PHASE ONLY: implement the complete real-acceptance runner/config and meaningful OFFLINE tests, commit locally, leave EXP-008 passes=false, and STOP with exact invocation/evidence. Do NOT create/start the persistent acceptance ledger and do NOT send any real model-generation request. The supervisor will review the committed runner, then execute the one real acceptance itself. Coding CLI usage is separate. Live mode must refuse before ledger/auth/model use unless an ignored supervisor review file binds the current implementation commit; offline/preflight mode must not require or create a live ledger. Do not manufacture that review file. Preparation ending with EXP008 false is expected, not a failed story."
try: result=subprocess.run([str(Path.home()/'.kimi-code/bin/kimi'),'-p',prompt],timeout=7200)
except subprocess.TimeoutExpired:
 failures=(old.get('failures',0) if old.get('story')==sid else 0)+1
 state.write_text(json.dumps(dict(story=sid,failures=failures,updated=time.time(),reason='programming_timeout')))
 print('STOP: programming iteration timeout; recorded failed attempt',sid);sys.exit(4)
now=json.loads((r/'prd.json').read_text()); passed=next(s for s in now['userStories'] if s['id']==sid)['passes']
if sid == 'EXP-008' and not passed and result.returncode == 0:
 state.write_text(json.dumps(dict(story=sid,failures=0,updated=time.time(),reason='awaiting_runner_supervisor_review')))
 print('EXP-008 preparation finished; supervisor must review and execute real acceptance; story remains incomplete')
 sys.exit(0)
failures=0 if passed else (old.get('failures',0) if old.get('story')==sid else 0)+1
state.write_text(json.dumps(dict(story=sid,failures=failures,updated=time.time())))
if not passed:print('STOP: story remains failing',sid);sys.exit(5)
if result.returncode:sys.exit(result.returncode)
