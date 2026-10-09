"""Cold auth-helper startup in the actual clean-PATH deployment shape."""
from pathlib import Path

from research.search.credentials import KimiAuthHelper


def test_cold_helper_finds_configured_home_binary_when_path_has_no_kimi(tmp_path, monkeypatch):
    binary = tmp_path / '.kimi-code/bin/kimi'
    binary.parent.mkdir(parents=True)
    binary.write_text('synthetic executable never actually launched')
    binary.chmod(0o700)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr('research.search.credentials.shutil.which', lambda name: None)
    launched = []
    class FakeProcess:
        pid = 12345
        terminated = False
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
        def kill(self):
            self.terminated = True
    process = FakeProcess()
    def spawn(argv, **kwargs):
        launched.append((argv, kwargs))
        return process
    helper = KimiAuthHelper(spawn_fn=spawn)
    monkeypatch.setattr(helper, '_healthy', lambda port: True)
    monkeypatch.setattr(helper, '_free_port', lambda: 32123)
    try:
        entry = helper.ensure()
        assert entry['owned'] is True
        assert launched[0][0][0] == str(binary)
        assert '--no-open' in launched[0][0]
    finally:
        helper.close()
