"""Session tests: these spawn real worker subprocesses, so they are the slow ones."""

import asyncio

import pytest

from jaxplorer.session import WorkerSession

HANGS = "import jax\nwhile True:\n    pass\n"


@pytest.fixture
async def session():
    session = WorkerSession(platform="cpu")
    try:
        yield session
    finally:
        await session.close()


async def test_handshake_reports_the_backend(session):
    info = await session.start()

    assert info.platform == "cpu"
    assert info.jax_version
    assert info.devices
    assert "cpu" in info.summary()


async def test_compile_roundtrip(session, snippet):
    result = await session.compile(snippet)

    assert result.fatal is None
    assert result.stages["optimized_hlo"].ok


async def test_a_burst_of_requests_collapses_to_the_newest(session, snippet):
    await session.start()  # keep startup cost out of the burst

    results = await asyncio.gather(*(session.compile(snippet) for _ in range(4)))

    # Everything queued behind the in-flight request is dropped except the newest, so at
    # most two compiles actually happen: the one already running and the final one.
    delivered = [r for r in results if r is not None]
    assert len(delivered) <= 2
    assert results[-1] is not None
    assert all(r.fatal is None for r in delivered)


async def test_timeout_kills_the_worker_and_the_session_recovers(session, snippet):
    session.timeout = 5.0

    result = await session.compile(HANGS)
    assert result is not None
    assert "timed out" in result.fatal
    assert session.info is None  # the worker was killed

    recovered = await session.compile(snippet)
    assert recovered is not None
    assert recovered.fatal is None
    assert recovered.stages["jaxpr"].ok


async def test_a_response_larger_than_the_pipe_line_limit_survives(session):
    # 60 layers is an ordinary model and its IR runs well past asyncio's 64 KiB readline
    # limit, which is why responses are length-prefixed rather than newline-delimited.
    source = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "def f(x):\n"
        "    for i in range(60):\n"
        "        x = jnp.tanh(x @ jnp.full((32, 32), 0.01 * (i + 1))) + i\n"
        "    return x.sum()\n"
        "args = (jax.ShapeDtypeStruct((32, 32), jnp.float32),)\n"
    )
    result = await session.compile(source)

    assert result is not None
    assert result.fatal is None
    hlo = result.stages["optimized_hlo"].text
    assert hlo is not None
    assert len(result.to_json()) > 64 * 1024
    assert "ENTRY" in hlo


async def test_cancelling_a_compile_during_startup_leaves_a_usable_session(snippet):
    session = WorkerSession(platform="cpu")
    try:
        task = asyncio.create_task(session.compile(snippet))
        await asyncio.sleep(0.05)  # spawned, still importing jax
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # A half-started worker must not wedge the session, since a keystroke during the
        # first boot is the most ordinary thing in the world.
        result = await session.compile(snippet)
        assert result is not None
        assert result.fatal is None
        assert result.stages["jaxpr"].ok
    finally:
        await session.close()


async def test_restart_brings_up_a_fresh_worker(session, snippet):
    first = await session.start()
    second = await session.restart()

    assert second.platform == first.platform
    result = await session.compile(snippet)
    assert result.fatal is None


async def test_snippet_stdout_does_not_corrupt_the_protocol(session):
    source = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "print('hello from the snippet')\n"
        "def f(x):\n"
        "    return x + 1\n"
        "args = (jnp.arange(3.0),)\n"
    )
    result = await session.compile(source)

    assert result is not None
    assert result.fatal is None
    assert result.stages["jaxpr"].ok
    # The print landed on stderr, where it is available for diagnostics.
    await asyncio.sleep(0.1)
    assert "hello from the snippet" in session.stderr_tail()


async def test_unavailable_platform_fails_to_start():
    session = WorkerSession(platform="definitely-not-a-backend")
    try:
        result = await session.compile("def f(x):\n    return x\nargs = (1.0,)\n")
        assert result is not None
        assert result.fatal
    finally:
        await session.close()
