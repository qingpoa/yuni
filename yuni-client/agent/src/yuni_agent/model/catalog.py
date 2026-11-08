from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import httpx

from yuni_agent.model.client import ModelClientError


def v1_endpoint(base_url: str, resource: str) -> str:
    parts = urlsplit(base_url.rstrip('/'))
    path = parts.path.rstrip('/')
    if not path.endswith('/v1'):
        path += '/v1'
    return urlunsplit((parts.scheme, parts.netloc, f'{path}/{resource}', '', ''))


async def list_models(base_url: str, api_key: str) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
            response = await client.get(v1_endpoint(base_url, 'model'), headers={'Authorization': f'Bearer {api_key}'})
            if response.status_code in {404, 405}:
                response = await client.get(v1_endpoint(base_url, 'models'), headers={'Authorization': f'Bearer {api_key}'})
        if response.status_code in {401, 403}:
            raise ModelClientError('模型列表鉴权失败，请检查 API Key')
        if response.status_code >= 400:
            raise ModelClientError(f'获取模型列表失败（HTTP {response.status_code}）')
        if len(response.content) > 1024 * 1024:
            raise ModelClientError('模型列表数据过大')
        payload = response.json()
        rows = payload.get('data', payload.get('models')) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or len(rows) > 10000:
            raise ValueError
        models = set()
        for row in rows:
            name = row.get('id', row.get('name')) if isinstance(row, dict) else row
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
                raise ValueError
            models.add(name.strip())
        return sorted(models, key=lambda name: (name.casefold(), name))
    except httpx.TimeoutException:
        raise ModelClientError('获取模型列表超时，请重试') from None
    except httpx.HTTPError:
        raise ModelClientError('模型列表连接失败，请重试') from None
    except ValueError:
        raise ModelClientError('模型列表格式无效') from None
