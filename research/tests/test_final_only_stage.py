"""Offline guards for the separately authorized final-only amendment."""
import json
from pathlib import Path
import pytest
from research.comparison.final_only_stage import StageJournal, verify_original

def test_exclusive_stage_and_intents_prevent_relaunch_and_duplicates(tmp_path: Path):
    roster=[f'cell-{i}' for i in range(17)]
    stage=tmp_path/'stage'
    journal=StageJournal(stage,roster,now=lambda:1000.,monotonic=lambda:10.)
    for key in roster: journal.reserve(key,{'cell_key':key})
    assert len(list((stage/'final').glob('*/intent.json')))==17
    with pytest.raises(RuntimeError): journal.reserve(roster[0],{})
    with pytest.raises(RuntimeError): journal.reserve('extra',{})
    with pytest.raises(FileExistsError): StageJournal(stage,roster)
    assert len(list((stage/'final').glob('*/intent.json')))==17

def test_monotonic_deadline_and_stop_refuse_further_submission(tmp_path: Path):
    clock=[1000.,10.]
    j=StageJournal(tmp_path/'stage',[str(i) for i in range(17)],now=lambda:clock[0],monotonic=lambda:clock[1])
    clock[1]+=14401
    with pytest.raises(RuntimeError): j.reserve('0',{})
    assert not (tmp_path/'stage/final').exists()
    clock[1]=10.
    j.stop('cleanup_unproven')
    with pytest.raises(RuntimeError): j.reserve('1',{})
    assert json.loads((tmp_path/'stage/launch.json').read_text())['deadline_epoch']==15400.

def test_original_hash_guard_rejects_changes_and_additions(tmp_path: Path):
    import hashlib
    original=tmp_path/'.runtime/comparison-istratde'
    original.mkdir(parents=True)
    p=original/'winner.py';p.write_text('original')
    audit={'original_runtime_file_hashes':{'winner.py':hashlib.sha256(p.read_bytes()).hexdigest()}}
    verify_original(tmp_path,audit)
    p.write_text('changed')
    with pytest.raises(ValueError,match='original_artifact_changed'): verify_original(tmp_path,audit)
    p.write_text('original');(original/'unexpected').write_text('extra')
    with pytest.raises(ValueError,match='roster_changed'): verify_original(tmp_path,audit)
