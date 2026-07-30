"""Auth tests for the graph endpoints.

Need no OpenAI key and no database, so they run in the default `make test`. Nothing asserts a
successful response: the app is built without its lifespan, so anything past auth reaches a
handler with no Graphiti client and fails. Only whether require_api_key let the request through
is asserted — see _assert_passed_auth.

The real graph_service.main is reloaded per case rather than a stand-in assembled, because its
wiring is what these guard: the likeliest regression is a router included without the auth
dependency, and only the real module can catch that.
"""

import importlib
import time

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError

from graph_service import auth, config
from graph_service.auth import MAX_FAILED_AUTH, require_api_key
from graph_service.config import MIN_API_KEY_LENGTH

SECRET = 'test-api-key-6Yp2Qk'
assert len(SECRET) >= MIN_API_KEY_LENGTH, 'the fixture key must itself be a valid key'

# The allowlist the app is held to; every other route must carry the auth dependency. An
# allowlist, not protected prefixes, so an unpredicted route name fails by default. The docs are
# deliberately public — see main.py.
PUBLIC_PATHS = {'/healthcheck', '/docs', '/docs/oauth2-redirect', '/redoc', '/openapi.json'}
DOCS_PATHS = ['/openapi.json', '/docs', '/redoc']


@pytest.fixture(autouse=True)
def _fresh_rejection_budget():
    """Empty the failed-auth budget around every case.

    Autouse, not optional: the budget lives in graph_service.auth, which build_app does not
    reload, so rejections would accumulate module-wide. This file provokes more than
    MAX_FAILED_AUTH of them, so cases asserting 401 would see 429 — order-dependently, which a
    single -k run would hide.
    """
    auth._recent_rejections.clear()
    yield
    auth._recent_rejections.clear()


@pytest.fixture
def build_app(monkeypatch):
    """Return a factory that builds the real app with GRAPHITI_API_KEY set to a given value.

    The process environment should be the only input, and two things otherwise get a say:
    Settings reads env_file='.env' relative to the cwd (server/ under `make test`), and
    graphiti_core calls load_dotenv() on import, which searches upward from server/.venv and
    finds the same file whatever the cwd. Either hands a developer with a server/.env the very
    key test_no_key_means_no_service asserts the absence of, failing the suite on their machine
    only. So: no env file for Settings, and the environment is set after the import that may
    inject one.
    """
    monkeypatch.setitem(config.Settings.model_config, 'env_file', None)

    def _build(graphiti_api_key: str | None):
        # Imported before the environment is arranged, because this is the import that
        # triggers load_dotenv() — on the first call, which is the one that matters. Nothing
        # reads settings yet: the app is assembled at import, get_settings() runs in lifespan.
        import graph_service.main as main

        monkeypatch.setenv('OPENAI_API_KEY', 'unused-by-these-tests')
        if graphiti_api_key is None:
            monkeypatch.delenv('GRAPHITI_API_KEY', raising=False)
        else:
            monkeypatch.setenv('GRAPHITI_API_KEY', graphiti_api_key)
        # Both cached at import time, so both must be discarded for a new key to take effect.
        config.get_settings.cache_clear()
        return importlib.reload(main).app

    yield _build
    config.get_settings.cache_clear()


def _search(app, headers=None):
    client = TestClient(app, raise_server_exceptions=False)
    return client.post(
        '/search', json={'query': 'anything', 'group_ids': ['g']}, headers=headers or {}
    )


def _assert_passed_auth(response):
    """Assert only that require_api_key let the request through.

    Not `== 500`: what the handler does next is not this file's business, and pinning it would
    break the day the fixture gains a lifespan. 401 is the one status meaning auth rejected it.
    """
    assert response.status_code != 401, response.text


@pytest.mark.parametrize('graphiti_api_key', [None, '', '   '], ids=['unset', 'blank', 'space'])
def test_no_key_means_no_service(build_app, graphiti_api_key):
    """Auth is mandatory, so a missing key is a startup error and not an open API.

    A service that boots without a key and one that 401s everything both look healthy to Render,
    and only one is safe. Blank and whitespace count as missing: a stray `GRAPHITI_API_KEY=` must
    not become a key of '' or ' ' that the deployment accepts.

    Asserted against the lifespan rather than get_settings() alone, because that is what makes it
    a startup failure — uvicorn runs it before serving, so the process exits. Entering it is
    enough, since settings are read on the first line.
    """
    app = build_app(graphiti_api_key)
    with pytest.raises(ValidationError, match='graphiti_api_key'), TestClient(app):
        pass  # never reached: entering the lifespan is what raises


def test_a_guessable_key_is_refused_at_startup(build_app):
    """A key too short to survive guessing fails the deploy, rather than serving traffic.

    The budget stops a stranger working through the keyspace, not a key already at the front of
    it. Rotation is a hand edit in the Dashboard, and nothing there would reject
    `GRAPHITI_API_KEY=dev` — this is what does. One character under the floor, so the boundary is
    pinned rather than a value that would pass a floor set lower by accident.
    """
    app = build_app('k' * (MIN_API_KEY_LENGTH - 1))
    with pytest.raises(ValidationError, match='string_too_short'), TestClient(app):
        pass  # never reached: entering the lifespan is what raises


def test_a_key_at_the_length_floor_is_accepted(build_app):
    """The other half of the boundary: the floor is inclusive, so it can't drift upward."""
    key = 'k' * MIN_API_KEY_LENGTH
    app = build_app(key)
    _assert_passed_auth(_search(app, {'Authorization': f'Bearer {key}'}))


def test_configured_key_accepts_the_right_bearer_token(build_app):
    app = build_app(SECRET)
    _assert_passed_auth(_search(app, {'Authorization': f'Bearer {SECRET}'}))


@pytest.mark.parametrize(
    'headers',
    [
        pytest.param({}, id='no-header'),
        pytest.param({'Authorization': 'Bearer wrong-key'}, id='wrong-key'),
        pytest.param({'Authorization': 'Bearer '}, id='empty-token'),
        pytest.param({'Authorization': f'Basic {SECRET}'}, id='wrong-scheme'),
        pytest.param({'Authorization': SECRET}, id='bare-token-no-scheme'),
        pytest.param({'X-API-Key': SECRET}, id='wrong-header'),
        # The right value one byte short: guards against compare_digest becoming a prefix
        # comparison.
        pytest.param({'Authorization': f'Bearer {SECRET[:-1]}'}, id='truncated-key'),
        # Raw bytes, since httpx refuses a non-ASCII str header. Starlette decodes as
        # latin-1, so this arrives as a non-ASCII str, which compare_digest refuses to
        # compare. Must be a 401, not a 500 for anyone sending a stray high byte.
        pytest.param({b'Authorization': b'Bearer \xff'}, id='non-ascii-token'),
    ],
)
def test_configured_key_rejects_everything_else(build_app, headers):
    app = build_app(SECRET)
    assert _search(app, headers).status_code == 401


# Every value is over MIN_API_KEY_LENGTH, and the match below pins the pattern error: a short
# non-ASCII key would fail on length and pass without the ASCII rule existing at all.
@pytest.mark.parametrize(
    'graphiti_api_key',
    [
        pytest.param('clé-secrète-assez-longue', id='accent'),
        pytest.param('key-with-an-emoji-🔑', id='emoji'),
        # A tab survives RequiredStr's strip only mid-value, and a control character in a
        # header is malformed however it is encoded.
        pytest.param('key\twith\ttabs\tin\tit', id='control-char'),
    ],
)
def test_a_non_ascii_key_is_refused_at_startup(build_app, graphiti_api_key):
    """A key that can't survive an HTTP header fails the deploy, rather than every request.

    Nothing in the Dashboard rejects a passphrase on rotation. Such a key authenticates for a
    client encoding the header as latin-1 and not for one using UTF-8, so it can't be relied on
    either way — and the operator who set it sees the same 401 as someone who mistyped it.
    Failing at startup names the actual problem.
    """
    app = build_app(graphiti_api_key)
    with pytest.raises(ValidationError, match='string_pattern_mismatch'), TestClient(app):
        pass  # never reached: entering the lifespan is what raises


def test_a_burst_of_wrong_keys_is_rate_limited(build_app):
    """The key can't be brute-forced, and the endpoint isn't a free scanner target.

    Exactly MAX_FAILED_AUTH rejections are spent first, pinning the boundary both ways: the last
    inside the budget still gets its 401, the first past it does not.
    """
    app = build_app(SECRET)
    wrong = {'Authorization': 'Bearer wrong-key-but-long-enough'}
    for attempt in range(MAX_FAILED_AUTH):
        assert _search(app, wrong).status_code == 401, f'budgeted attempt {attempt} was refused'

    limited = _search(app, wrong)
    assert limited.status_code == 429
    # Conventional on a 429, and the client has no other way to know how long to wait.
    assert 0 < int(limited.headers['Retry-After']) <= auth.FAILED_AUTH_WINDOW_SECONDS


def test_the_right_key_is_never_rate_limited(build_app):
    """Why a global budget is safe: it can't lock a real client out.

    Otherwise one stranger guessing keys takes the API down for whoever holds the real one — a
    worse outcome than the brute-force the budget exists to stop.
    """
    app = build_app(SECRET)
    wrong = {'Authorization': 'Bearer wrong-key-but-long-enough'}
    for _ in range(MAX_FAILED_AUTH + 5):
        _search(app, wrong)
    assert _search(app, wrong).status_code == 429, 'the budget should be spent by now'

    _assert_passed_auth(_search(app, {'Authorization': f'Bearer {SECRET}'}))


def test_the_budget_recovers_once_the_window_passes(build_app):
    """A spent budget is a delay, not a latch — otherwise one burst closes the API for good.

    The window is moved rather than waited out: back-dating the deque's monotonic timestamps is
    the same thing to the code and keeps the suite fast.
    """
    app = build_app(SECRET)
    wrong = {'Authorization': 'Bearer wrong-key-but-long-enough'}
    for _ in range(MAX_FAILED_AUTH):
        _search(app, wrong)
    assert _search(app, wrong).status_code == 429

    stale = time.monotonic() - auth.FAILED_AUTH_WINDOW_SECONDS - 1
    auth._recent_rejections.extend([stale] * MAX_FAILED_AUTH)
    assert _search(app, wrong).status_code == 401


def test_rejection_names_the_scheme_to_retry_with(build_app):
    """RFC 9110 requires WWW-Authenticate on a 401, and it tells clients what to send."""
    app = build_app(SECRET)
    assert _search(app).headers.get('WWW-Authenticate') == 'Bearer'


def test_healthcheck_stays_public(build_app):
    """If auth ever covers it, every deploy fails its health check and rolls back."""
    app = build_app(SECRET)
    response = TestClient(app).get('/healthcheck')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


@pytest.mark.parametrize('path', DOCS_PATHS)
def test_the_docs_are_browsable(build_app, path):
    """Deliberately public, and worth a test because it is a judgement call, not a default.

    A browser can't attach a bearer header to a navigation, so protecting these would make them
    unreachable by clicking a link on any deployment with a key.
    """
    app = build_app(SECRET)
    assert TestClient(app).get(path).status_code == 200


def test_the_docs_offer_the_bearer_scheme(build_app):
    """Swagger's Authorize button, which keeps the docs usable against a deployment.

    It comes from the routers' dependency, easy to lose by protecting the routes some other way,
    and its absence would leave Try it out silently 401ing.
    """
    schema = TestClient(build_app(SECRET)).get('/openapi.json').json()
    assert schema['components']['securitySchemes']['HTTPBearer']['scheme'] == 'bearer'
    assert schema['paths']['/search']['post']['security'] == [{'HTTPBearer': []}]


def _is_protected(route) -> bool:
    """Whether require_api_key runs before this route's handler.

    Identity, not a name comparison, so a same-named dependency elsewhere can't satisfy it.
    Anything that is not an APIRoute has no dependant and counts as unprotected — a Mount or
    WebSocketRoute is exactly what this must not wave through.
    """
    if not isinstance(route, APIRoute):
        return False
    return any(dependency.call is require_api_key for dependency in route.dependant.dependencies)


def test_every_route_but_the_public_ones_is_protected(build_app):
    """The whole surface, not a list of prefixes someone has to remember to extend.

    A router included without the dependency, or a public endpoint added by accident, fails
    here; adding a route to PUBLIC_PATHS is then a deliberate edit.
    """
    app = build_app(SECRET)
    unprotected = {
        route.path
        for route in app.routes
        if route.path not in PUBLIC_PATHS and not _is_protected(route)
    }
    assert not unprotected, f'routes missing auth: {sorted(unprotected)}'
