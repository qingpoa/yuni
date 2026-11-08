from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from yuni_agent.api.app import create_app
from yuni_agent.config import StartupConfig
from yuni_agent.model.auth import AuthFileError, ModelAuth, load_auth, save_auth
from yuni_agent.runtime import AgentRuntime

KEY = 'test-key-not-real'


def make_client(root: Path):
    runtime = AgentRuntime(config=StartupConfig.for_test(root))
    return TestClient(create_app(runtime)), runtime, {
        'X-Yuni-Session': runtime.config.session_token.get_secret_value(),
    }


def model_input(**changes):
    return {
        'provider': 'openai_compatible', 'modelName': 'deepseek-flash',
        'baseUrl': 'https://api.deepseek.com', 'apiKey': KEY, **changes,
    }


def test_model_list_uses_saved_key_and_never_persists_draft(tmp_path, monkeypatch):
    from yuni_agent.api.routes import settings
    calls = []
    async def catalog(url, key):
        calls.append((url, key))
        return ['chat', 'jev-latest']
    monkeypatch.setattr(settings, 'list_models', catalog)
    client, runtime, headers = make_client(tmp_path)
    with client:
        assert client.post('/api/settings/models', json={}).status_code == 401
        assert client.post('/api/settings/models', headers=headers, json={}).status_code == 400
        client.put('/api/settings/model', headers=headers, json=model_input(baseUrl='https://gateway.example/v1'))
        before = runtime.auth_path.read_bytes()
        response = client.post('/api/settings/models', headers=headers, json={})
        assert response.json()['data'] == ['chat', 'jev-latest'] and KEY not in response.text
        assert calls == [('https://gateway.example/v1', KEY)]
        assert client.post('/api/settings/models', headers=headers, json={'baseUrl': 'https://other.example/v1'}).status_code == 400
        assert len(calls) == 1
        response = client.post('/api/settings/models', headers=headers, json={'baseUrl': 'http://new.example', 'apiKey': 'draft-key'})
        assert response.status_code == 200 and calls[-1] == ('http://new.example', 'draft-key')
        assert runtime.auth_path.read_bytes() == before
        for url in ['file:///tmp', 'https://name:password@example.com']:
            assert client.post('/api/settings/models', headers=headers, json={'baseUrl': url}).status_code == 422


def test_model_list_upstream_error_returns_safe_retryable_error(tmp_path, monkeypatch):
    from yuni_agent.api.routes import settings
    from yuni_agent.model.client import ModelClientError
    async def catalog(url, key):
        raise ModelClientError('获取模型列表超时，请重试')
    monkeypatch.setattr(settings, 'list_models', catalog)
    client, _, headers = make_client(tmp_path)
    with client:
        response = client.post('/api/settings/models', headers=headers, json={'baseUrl': 'https://gateway.example', 'apiKey': KEY})
        assert response.status_code == 502 and response.json()['code'] == 40303
        assert KEY not in response.text


def test_saved_auth_restores_after_restart_and_is_not_exported(tmp_path):
    client, runtime, headers = make_client(tmp_path)
    with client:
        saved = client.put('/api/settings/model', headers=headers, json=model_input())
        assert saved.status_code == 200
        assert saved.json()['data']['model']['configured'] is True
        assert json.loads(runtime.auth_path.read_text()) == {
            'model': 'deepseek-flash', 'url': 'https://api.deepseek.com', 'apikey': KEY,
        }
        assert KEY not in saved.text
        assert KEY not in client.get('/api/settings', headers=headers).text
        assert KEY not in client.get('/api/data/export', headers=headers).text
    restarted, restored, headers = make_client(tmp_path)
    with restarted:
        summary = restarted.get('/api/settings', headers=headers).json()['data']['model']
        assert summary['modelName'] == 'deepseek-flash'
        assert summary['baseUrl'] == 'https://api.deepseek.com'
        assert summary['configured'] is True
        assert restored.model_api_key.get_secret_value() == KEY


def test_empty_key_preserves_saved_key(tmp_path):
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.put('/api/settings/model', headers=headers, json=model_input())
        updated = client.put('/api/settings/model', headers=headers, json=model_input(
            modelName='deepseek-v4-pro', apiKey='  ', baseUrl='https://api.deepseek.com/v1/',
        ))
        assert updated.status_code == 200
        auth = load_auth(runtime.auth_path)
        assert auth.model == 'deepseek-v4-pro'
        assert auth.url == 'https://api.deepseek.com/v1'
        assert auth.apikey.get_secret_value() == KEY


def test_deleting_key_does_not_restore_it_on_restart(tmp_path):
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.put('/api/settings/model', headers=headers, json=model_input())
        response = client.delete('/api/settings/model/key', headers=headers)
        assert response.json()['data']['model']['configured'] is False
        assert json.loads(runtime.auth_path.read_text())['apikey'] == ''
    restarted, _, headers = make_client(tmp_path)
    with restarted:
        model = restarted.get('/api/settings', headers=headers).json()['data']['model']
        assert model['configured'] is False
        assert model['modelName'] == 'deepseek-flash'


@pytest.mark.parametrize('operation', ['clear', 'import'])
def test_clear_or_import_removes_auth_file(tmp_path, operation):
    client, runtime, headers = make_client(tmp_path)
    with client:
        backup = client.get('/api/data/export', headers=headers).content
        client.put('/api/settings/model', headers=headers, json=model_input())
        if operation == 'clear':
            response = client.delete('/api/data', headers=headers)
        else:
            response = client.post('/api/data/import', headers=headers,
                                   files={'file': ('backup.json', backup, 'application/json')})
        assert response.status_code == 200
        assert not runtime.auth_path.exists()
        assert runtime.model_api_key is None
    restarted, _, headers = make_client(tmp_path)
    with restarted:
        assert restarted.get('/local/bootstrap', headers=headers).json()['data']['modelConfigured'] is False


def test_auth_file_is_authoritative_for_model_and_url(tmp_path):
    config = StartupConfig.for_test(tmp_path)
    config.prepare_directories()
    save_auth(config.data_dir / 'auth.json', ModelAuth(
        model='manual-model', url='https://api.example.com/v1', apikey=KEY,
    ))
    client, runtime, headers = make_client(tmp_path)
    with client:
        model = client.get('/api/settings', headers=headers).json()['data']['model']
        assert model['modelName'] == 'manual-model'
        assert model['baseUrl'] == 'https://api.example.com/v1'
        assert model['configured'] is True


@pytest.mark.parametrize('payload', ['{invalid-json', json.dumps({
    'model': 'test', 'url': 'ftp://gateway.example.test', 'apikey': KEY,
})])
def test_invalid_auth_is_nonfatal_and_does_not_leak_secret(tmp_path, caplog, payload):
    config = StartupConfig.for_test(tmp_path)
    config.prepare_directories()
    (config.data_dir / 'auth.json').write_text(payload)
    client, runtime, headers = make_client(tmp_path)
    with client:
        assert client.get('/local/health', headers=headers).status_code == 200
        assert runtime.model_api_key is None
        assert KEY not in caplog.text
        assert (config.data_dir / 'auth.json').read_text() == payload


def test_atomic_replace_failure_preserves_file(tmp_path, monkeypatch):
    path = tmp_path / 'auth.json'
    original = ModelAuth(model='old', url='https://api.example.com', apikey=KEY)
    save_auth(path, original)

    def fail_replace(*args):
        raise OSError('write failed')

    monkeypatch.setattr('yuni_agent.model.auth.os.replace', fail_replace)
    with pytest.raises(AuthFileError):
        save_auth(path, ModelAuth(model='new', url=original.url, apikey='new-test-key'))
    assert load_auth(path).model == 'old'
    assert load_auth(path).apikey.get_secret_value() == KEY
    assert not list(tmp_path.glob('.auth-*.tmp'))


def test_save_failure_keeps_previous_runtime_and_database(tmp_path, monkeypatch):
    client, runtime, headers = make_client(tmp_path)
    with client:
        client.put('/api/settings/model', headers=headers, json=model_input())

        def reject(*args):
            raise AuthFileError(KEY)

        monkeypatch.setattr('yuni_agent.runtime.save_auth', reject)
        response = client.put('/api/settings/model', headers=headers, json=model_input(
            modelName='new-model', apiKey='new-test-key',
        ))
        assert response.status_code == 500
        assert KEY not in response.text
        assert runtime.model_api_key.get_secret_value() == KEY
        assert client.get('/api/settings', headers=headers).json()['data']['model']['modelName'] == 'deepseek-flash'


@pytest.mark.parametrize('base_url', [
    'http://127.0.0.1:11434/v1',
    'http://localhost:8080/v1',
    'http://[::1]:8080/v1',
    '  http://192.168.1.20:8080/v1/  ',
    'http://gateway.example.test:8080/v1',
])
def test_http_model_url_saves_and_restores(tmp_path, base_url):
    normalized = base_url.strip().rstrip('/')
    client, runtime, headers = make_client(tmp_path)
    with client:
        response = client.put('/api/settings/model', headers=headers, json=model_input(baseUrl=base_url))
        assert response.status_code == 200
        assert response.json()['data']['model']['baseUrl'] == normalized
        assert KEY not in response.text
        auth = load_auth(runtime.auth_path)
        assert auth is not None and auth.url == normalized
        assert auth.apikey.get_secret_value() == KEY
    restarted, restored, headers = make_client(tmp_path)
    with restarted:
        model = restarted.get('/api/settings', headers=headers).json()['data']['model']
        assert model['baseUrl'] == normalized
        assert model['configured'] is True
        assert restored.model_api_key.get_secret_value() == KEY


@pytest.mark.parametrize('base_url', [
    'ftp://gateway.example.test/v1',
    'file:///tmp/model',
    'http:///v1',
    '//gateway.example.test/v1',
    'http://user:password@gateway.example.test/v1',
    'https://user@gateway.example.test/v1',
])
def test_invalid_model_url_does_not_replace_saved_config(tmp_path, base_url):
    client, runtime, headers = make_client(tmp_path)
    with client:
        saved = client.put('/api/settings/model', headers=headers, json=model_input())
        assert saved.status_code == 200
        invalid = client.put('/api/settings/model', headers=headers, json=model_input(baseUrl=base_url))
        assert invalid.status_code == 422
        assert KEY not in invalid.text
        auth = load_auth(runtime.auth_path)
        assert auth is not None and auth.url == 'https://api.deepseek.com'
        assert auth.apikey.get_secret_value() == KEY
