import asyncio

from ingestion.deriv_client import DerivClient


def make_client():
    return DerivClient(
        app_id="1089", api_token="tok", ws_url="wss://example.invalid",
        options_token_url="https://example.invalid", account_id="ACC1",
    )


async def _run_recovers_after_transient_failures():
    client = make_client()
    attempts = []

    async def flaky_connect():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("network still down")
        # simulate what a real connect() does, minus the network
        client._ws = object()

    resubscribed = []

    async def fake_resubscribe_all():
        resubscribed.append(1)

    client.connect = flaky_connect
    client._resubscribe_all = fake_resubscribe_all

    sleeps = []
    orig_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        sleeps.append(seconds)
        await orig_sleep(0)

    asyncio.sleep = fast_sleep
    try:
        await client._recover_from_recv_pump_crash(ConnectionError("original drop"), max_attempts=5)
    finally:
        asyncio.sleep = orig_sleep

    assert len(attempts) == 3          # failed twice, succeeded on the 3rd
    assert resubscribed == [1]         # resubscribed exactly once, after success
    assert len(sleeps) == 2            # slept between the two failed attempts, not after success


def test_recv_pump_crash_recovers_after_transient_reconnect_failures():
    """Regression test for a live deployment: a websockets ConnectionClosedError
    inside _recv_pump used to just end the task with nothing to replace it --
    every symbol_worker's `await queue.get()` would then hang forever with no
    more ticks ever arriving, and the process would look 'alive' while doing
    nothing. The recv pump must now reconnect (with retry/backoff) and
    resubscribe every symbol itself before giving up."""
    asyncio.run(_run_recovers_after_transient_failures())


async def _run_gives_up_after_max_attempts():
    client = make_client()
    attempts = []

    async def always_fails():
        attempts.append(1)
        raise ConnectionError("network permanently down")

    resubscribed = []

    async def fake_resubscribe_all():
        resubscribed.append(1)

    client.connect = always_fails
    client._resubscribe_all = fake_resubscribe_all

    orig_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await orig_sleep(0)

    asyncio.sleep = fast_sleep
    try:
        # must not raise -- a persistently-down network should be logged and
        # given up on, not propagate out of the crash handler
        await client._recover_from_recv_pump_crash(ConnectionError("original drop"), max_attempts=4)
    finally:
        asyncio.sleep = orig_sleep

    assert len(attempts) == 4
    assert resubscribed == []


def test_recv_pump_crash_recovery_gives_up_gracefully_after_max_attempts():
    asyncio.run(_run_gives_up_after_max_attempts())


async def _run_recv_pump_triggers_recovery_on_crash():
    client = make_client()
    recovered_with = []

    async def fake_recover(exc, max_attempts=5):
        recovered_with.append(exc)

    client._recover_from_recv_pump_crash = fake_recover

    class ExplodingIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionError("boom")

    client._ws = ExplodingIterator()
    client._closed = False
    await client._recv_pump()

    assert len(recovered_with) == 1
    assert isinstance(recovered_with[0], ConnectionError)


def test_recv_pump_calls_recovery_on_crash_when_not_closed():
    asyncio.run(_run_recv_pump_triggers_recovery_on_crash())


async def _run_recv_pump_skips_recovery_when_deliberately_closed():
    client = make_client()
    recovered_with = []

    async def fake_recover(exc, max_attempts=5):
        recovered_with.append(exc)

    client._recover_from_recv_pump_crash = fake_recover

    class ExplodingIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionError("boom, but we meant to close")

    client._ws = ExplodingIterator()
    client._closed = True  # close() was called deliberately
    await client._recv_pump()

    assert recovered_with == []


def test_recv_pump_does_not_reconnect_after_a_deliberate_close():
    asyncio.run(_run_recv_pump_skips_recovery_when_deliberately_closed())
