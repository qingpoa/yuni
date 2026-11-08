from functools import partial

import pytest

from yuni_agent.chat.queue import ConversationQueue


@pytest.fixture(autouse=True)
def immediate_api_batches(monkeypatch):
    """Keep unrelated API tests fast; queue timing tests use the real default."""
    monkeypatch.setattr(
        'yuni_agent.api.routes.chat.ConversationQueue',
        partial(ConversationQueue, quiet_seconds=0),
    )
