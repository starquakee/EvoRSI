import threading
import time
from research.contracts.budget import BudgetLedger, CancellationToken, ModelUsage, guarded_attempt


def test_guarded_transports_are_serial(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    counter_lock = threading.Lock()
    active = [0]
    peak = [0]
    errors = []
    def transport(request):
        with counter_lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        entered.set()
        release.wait(2)
        with counter_lock:
            active[0] -= 1
        return {}
    with BudgetLedger.open(tmp_path / 'budget.json') as ledger:
        def attempt():
            try:
                guarded_attempt(ledger, CancellationToken(), transport, {},
                                estimated_input_tokens=20, max_output_tokens=20,
                                usage_fn=lambda response: ModelUsage(1, 1))
            except Exception as error:
                errors.append(error)
        first = threading.Thread(target=attempt)
        second = threading.Thread(target=attempt)
        first.start()
        assert entered.wait(1)
        second.start()
        time.sleep(.1)
        release.set()
        first.join(2)
        second.join(2)
        assert not first.is_alive() and not second.is_alive()
        assert not errors
        assert ledger.snapshot()['committed']['requests'] == 2
    assert peak[0] == 1, 'model transports overlapped despite a shared ledger'


def test_usage_above_reservation_stops_and_records_actual(tmp_path):
    with BudgetLedger.open(tmp_path / 'budget.json') as ledger:
        reservation = ledger.reserve(5, 5)
        ledger.commit(reservation.reservation_id, ModelUsage(10, 1))
        snapshot = ledger.snapshot()
        assert snapshot['committed'] == {'requests': 1, 'tokens': 11}
        assert snapshot['stopped'] is not None
        assert snapshot['stopped']['reason'] == 'usage_exceeded_reservation'
