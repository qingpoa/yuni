import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select

from yuni_agent.config import StartupConfig
from yuni_agent.model.auth import ModelCredentials
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import Sticker

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stickers.py"
spec = importlib.util.spec_from_file_location("build_stickers", SCRIPT)
build = importlib.util.module_from_spec(spec)
sys.modules["build_stickers"] = build  # dataclasses resolve annotations via sys.modules
spec.loader.exec_module(build)

GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def label_ok(image):
    return build.normalize_label({
        "description": f"小猫{image.relative}", "text_in_image": "好耶", "emotion": "开心",
        "usage_hint": "对方分享好消息时", "tags": ["猫", "庆祝"], "unsuitable": False,
    })


def make_source(tmp_path):
    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "a.gif").write_bytes(GIF)
    (source / "sub" / "dup.gif").write_bytes(GIF)
    (source / "b.png").write_bytes(PNG)
    (source / "note.txt").write_text("not image")
    return source


@pytest.fixture
def data_dir(tmp_path):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path / "agent"))
    runtime.initialize_storage()
    runtime.close_storage()
    return runtime.config.data_dir


def test_scan_dedupes_and_rejects_non_images(tmp_path):
    images, skipped = build.scan_source(make_source(tmp_path))
    assert sorted(i.relative for i in images) == ["a.gif", "b.png"]
    assert any("重复" in s for s in skipped) and any("note.txt" in s for s in skipped)


def test_normalize_and_compose():
    label = build.normalize_label({"description": " 抱拳 ", "emotion": "未知", "tags": "x", "unsuitable": "yes"})
    assert label["emotion"] == "其他" and label["tags"] == [] and label["unsuitable"] is False
    with pytest.raises(build.LabelError):
        build.normalize_label({"description": ""})
    text = build.compose_description(label_ok(type("I", (), {"relative": "a"})()))
    assert "图中文字“好耶”" in text and "情绪：开心" in text and len(text) <= 500


def test_parse_json_object_handles_fences():
    assert build.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert build.parse_json_object('说明 {"a": 2} 尾巴') == {"a": 2}
    with pytest.raises(build.LabelError):
        build.parse_json_object("no json")


def test_label_resumes_retries_and_keeps_edits(tmp_path):
    source = make_source(tmp_path)
    calls = []

    def flaky(image):
        calls.append(image.relative)
        if image.relative == "b.png":
            raise build.LabelError("boom")
        return label_ok(image)

    stats = build.run_label(source, flaky, model_name="vl", log=lambda _: None)
    assert stats["labeled"] == 1 and stats["failed"] == 1 and calls.count("b.png") == 2
    manifest = json.loads((source / build.MANIFEST_NAME).read_text(encoding="utf-8"))
    items = {i["file"]: i for i in manifest["items"]}
    assert items["b.png"]["status"] == "failed" and items["a.gif"]["model"] == "vl"

    items["a.gif"].update(description="用户改过", edited=True)
    (source / build.MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    calls.clear()
    build.run_label(source, lambda i: calls.append(i.relative) or label_ok(i),
                    model_name="vl", relabel=True, log=lambda _: None)
    assert calls == ["b.png"]
    items = {i["file"]: i for i in json.loads((source / build.MANIFEST_NAME).read_text(encoding="utf-8"))["items"]}
    assert items["a.gif"]["description"] == "用户改过" and items["b.png"]["status"] == "labeled"


def test_unsuitable_is_disabled(tmp_path):
    source = make_source(tmp_path)
    build.run_label(source, lambda i: {**label_ok(i), "unsuitable": True}, model_name="vl", log=lambda _: None)
    items = json.loads((source / build.MANIFEST_NAME).read_text(encoding="utf-8"))["items"]
    assert all(i["enabled"] is False for i in items)


def test_import_inserts_updates_and_is_idempotent(tmp_path, data_dir):
    source = make_source(tmp_path)
    build.run_label(source, label_ok, model_name="vl", log=lambda _: None)
    assert build.run_import(source, data_dir, dry_run=True, log=lambda _: None)["inserted"] == 2
    assert not (data_dir / "stickers").exists()
    assert build.run_import(source, data_dir, log=lambda _: None)["inserted"] == 2
    assert build.run_import(source, data_dir, log=lambda _: None)["unchanged"] == 2

    manifest_path = source / build.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0].update(description="新描述", enabled=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (source / "b.png").write_bytes(PNG + b"changed")
    stats = build.run_import(source, data_dir, log=lambda _: None)
    assert stats["updated"] == 1 and stats["skipped"] == 1

    from yuni_agent.storage.database import Database
    database = Database.open(data_dir / "yuni.db")
    with database.session() as session:
        rows = {r.sha256: r for r in session.scalars(select(Sticker))}
    database.close()
    gif = rows[hashlib.sha256(GIF).hexdigest()]
    assert gif.relative_path.startswith("stickers/") and gif.content_type == "image/gif"
    assert (data_dir / gif.relative_path).read_bytes() == GIF
    changed = next(r for r in rows.values() if r.description == "新描述")
    assert changed.enabled is False


def fake_pack(items_override=None, bad_revision=False):
    md5 = hashlib.md5(GIF).hexdigest()
    items = items_override or [
        {"md5": md5, "sha256": hashlib.sha256(GIF).hexdigest(), "filename": f"emoticons/{md5}.gif",
         "format": "gif", "size": len(GIF), "caption": "原描述：小猫"},
        {"md5": "0" * 32, "sha256": "0" * 64, "filename": "emoticons/../../evil.gif", "format": "gif", "size": 1},
    ]
    raw = json.dumps({"schema_version": 1, "collection": "curated", "items": items}).encode()
    files = {
        "packs.json": json.dumps({"schema_version": 1, "packs": [{
            "id": "curated", "manifest": "packs/curated.json",
            "manifest_sha256": "f" * 64 if bad_revision else hashlib.sha256(raw).hexdigest()}]}).encode(),
        "packs/curated.json": raw,
        f"emoticons/{md5}.gif": GIF,
    }
    calls = []

    def fetch(url):
        calls.append(url)
        return files[url.removeprefix("https://pack.test/")]

    return fetch, calls


def test_fetch_pack_verifies_and_seeds_hint(tmp_path):
    fetch, calls = fake_pack()
    dest = tmp_path / "pack"
    stats = build.run_fetch_pack(dest, base_url="https://pack.test", fetch=fetch, log=lambda _: None)
    assert stats == {"downloaded": 1, "existing": 0, "skipped": 1}
    assert not any("evil" in c for c in calls) and not (tmp_path / "evil.gif").exists()
    manifest = json.loads((dest / build.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["items"][0]["source_caption"] == "原描述：小猫" and manifest["items"][0]["status"] == "pending"

    assert build.run_fetch_pack(dest, base_url="https://pack.test", fetch=fetch, log=lambda _: None)["existing"] == 1
    hints = []
    build.run_label(dest, lambda i: hints.append(i.hint) or label_ok(i), model_name="vl", log=lambda _: None)
    assert hints == ["原描述：小猫"]


def test_fetch_pack_rejects_revision_mismatch_and_corrupt_file(tmp_path):
    fetch, _ = fake_pack(bad_revision=True)
    with pytest.raises(SystemExit):
        build.run_fetch_pack(tmp_path / "a", base_url="https://pack.test", fetch=fetch, log=lambda _: None)
    md5 = hashlib.md5(GIF).hexdigest()
    fetch, _ = fake_pack([{"md5": md5, "sha256": "1" * 64, "filename": f"emoticons/{md5}.gif",
                           "format": "gif", "size": len(GIF)}])
    stats = build.run_fetch_pack(tmp_path / "b", base_url="https://pack.test", fetch=fetch, log=lambda _: None)
    assert stats["skipped"] == 1 and not list((tmp_path / "b").glob("*.gif"))


def test_import_with_source_captions(tmp_path, data_dir):
    fetch, _ = fake_pack()
    dest = tmp_path / "pack"
    build.run_fetch_pack(dest, base_url="https://pack.test", fetch=fetch, log=lambda _: None)
    # unlabeled pack items are importable with their own caption
    assert build.run_import(dest, data_dir, caption="source", log=lambda _: None)["inserted"] == 1
    manifest_path = dest / build.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0].update(status="labeled", description="模型乱标", enabled=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    assert build.run_import(dest, data_dir, caption="source", log=lambda _: None)["unchanged"] == 1
    manifest["items"][0].update(description="人工描述", edited=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    assert build.run_import(dest, data_dir, caption="source", log=lambda _: None)["updated"] == 1

    from yuni_agent.storage.database import Database
    database = Database.open(data_dir / "yuni.db")
    with database.session() as session:
        row = session.scalars(select(Sticker)).one()
        assert row.description == "人工描述" and row.enabled is False
    database.close()


def test_hint_is_sent_as_reference(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"description": "猫"}'}}]})

    real_client = httpx.Client
    monkeypatch.setattr(build.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler)))
    auth = ModelCredentials(model="vl", url="https://example.test/v1", apikey=SecretStr("k"))
    build.make_vision_labeler(auth)(build.ImageFile(Path("a.gif"), "a.gif", GIF, "image/gif", "x", "原描述"))
    text = seen["body"]["messages"][1]["content"][0]["text"]
    assert "原描述" in text and "以画面为准" in text


def test_normalize_semantics_uses_fixed_vocabulary():
    result = build.normalize_semantics({"emotions": ["调侃", "乱写", "调侃", "敷衍", "开心"], "usage": " 对方炫耀时 "})
    assert result == {"emotions": ["调侃", "敷衍"], "usage": "对方炫耀时", "uncertain": False}
    assert build.normalize_semantics({"emotions": ["乱写"], "usage": "x"}) == {"emotions": [], "usage": "", "uncertain": True}


def test_enrich_and_import_semantics(tmp_path, data_dir, monkeypatch):
    fetch, _ = fake_pack()
    dest = tmp_path / "pack"
    build.run_fetch_pack(dest, base_url="https://pack.test", fetch=fetch, log=lambda _: None)
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        content = json.dumps({"emotions": ["开心"], "usage": "分享好消息时"}, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    real_client = httpx.Client
    monkeypatch.setattr(build.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler)))
    auth = ModelCredentials(model="deepseek-chat", url="https://example.test/v1", apikey=SecretStr("k"))
    assert build.run_enrich(dest, build.make_text_enricher(auth), model_name="ds", log=lambda _: None)["enriched"] == 1
    assert "原描述：小猫" in seen[0]["messages"][1]["content"] and seen[0]["thinking"] == {"type": "disabled"}
    assert "image_url" not in json.dumps(seen[0])
    # already done → no new calls; edited semantics are kept even with --force
    assert build.run_enrich(dest, build.make_text_enricher(auth), model_name="ds", log=lambda _: None)["enriched"] == 0
    manifest_path = dest / build.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0]["semantics"].update(emotions=["撒娇"], edited=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    assert build.run_enrich(dest, build.make_text_enricher(auth), model_name="ds", force=True, log=lambda _: None)["enriched"] == 0

    assert build.run_import(dest, data_dir, caption="source", log=lambda _: None)["inserted"] == 1
    from yuni_agent.storage.database import Database
    database = Database.open(data_dir / "yuni.db")
    with database.session() as session:
        row = session.scalars(select(Sticker)).one()
        assert row.emotions == ["撒娇"] and row.usage == "分享好消息时"
    database.close()
    assert build.run_import(dest, data_dir, caption="source", log=lambda _: None)["unchanged"] == 1


def test_import_requires_database(tmp_path):
    source = make_source(tmp_path)
    with pytest.raises(SystemExit):
        build.run_import(source, tmp_path / "missing")


def test_vision_request_shape_and_auth_failure(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        content = json.dumps({"description": "猫", "emotion": "开心", "usage_hint": "开心时", "tags": ["猫"]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    real_client = httpx.Client
    monkeypatch.setattr(build.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler)))
    auth = ModelCredentials(model="vl", url="https://example.test/v1", apikey=SecretStr("k"))
    image = build.ImageFile(Path("a.gif"), "a.gif", GIF, "image/gif", "x")
    assert build.make_vision_labeler(auth)(image)["description"] == "猫"
    part = seen["body"]["messages"][1]["content"][1]
    assert part["image_url"]["url"].startswith("data:image/gif;base64,") and seen["auth"] == "Bearer k"

    monkeypatch.setattr(build.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(lambda r: httpx.Response(401))))
    with pytest.raises(SystemExit):
        build.make_vision_labeler(auth)(image)
