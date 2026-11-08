"""Image generation tool pack (`generate_image`)."""
from __future__ import annotations

import asyncio
import json

from sqlalchemy import select

from yuni_agent.model.image import (
    ASPECT_SIZES,
    MAX_REFERENCE_BYTES,
    ImageGenerationError,
    sniff_image_type,
    store_image,
)
from yuni_agent.tools.errors import ToolExecutionError
from yuni_agent.tools.models import GenerateImageArguments, ToolResult, ToolRisk
from yuni_agent.tools.registry import RegisteredTool, ToolContext, ToolPreview

GENERATE_IMAGE_DESCRIPTION = (
    "生成一张图片，图片会随你下一次 reply 发给用户。两种用途由 mode 区分："
    "mode=\"companion_photo\" 用于用户想看你的样子、你的自拍、你眼前/身边的照片——"
    "会尽量以你的头像为同一人；mode=\"creative\" 用于用户明确要你画别的东西"
    "（插画、风景、动物、壁纸等）。"
    "仅在用户本轮明确想要图片时调用；闲聊提到画面、问能不能画、用户只发了图片都不算。"
    "不画用户未要求的其他真实人物。每轮最多一次，失败时如实告诉用户。"
    "生图需要十几秒，可以先简短 reply 一句再调用；生成成功后必须 reply 递交，不能直接 stop。"
)


def _prepare_generate_image(context: ToolContext, _arguments: GenerateImageArguments) -> ToolPreview:
    if not getattr(context.runtime, "image_generation_ready", False):
        raise ToolExecutionError("生图功能未配置或已关闭")
    # Costs money and sends text to a third party, so it is not a read; it still
    # runs without a dialog because the user asked for the picture in this turn.
    return ToolPreview(summary="按用户要求生成图片", requires_confirmation=False)


def _companion_avatar_reference(runtime) -> tuple[bytes, str] | None:
    """Return (bytes, mime) of the companion's uploaded avatar, or None.

    Preset avatars are frontend color tiles with no backend pixels, so only an
    uploaded avatar can serve as an image-to-image reference.
    """
    from yuni_agent.storage.models import AvatarFile, Companion
    with runtime.require_database().session() as session:
        companion = session.scalar(select(Companion).limit(1))
        if companion is None or companion.avatar_type != "uploaded" or companion.avatar_file_id is None:
            return None
        avatar = session.get(AvatarFile, companion.avatar_file_id)
        if avatar is None:
            return None
        path = runtime.config.data_dir / avatar.relative_path
    try:
        if not path.is_file() or path.stat().st_size > MAX_REFERENCE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    sniffed = sniff_image_type(data)
    if sniffed is None:
        return None
    return data, sniffed


async def _generate_image(context: ToolContext, arguments: GenerateImageArguments) -> ToolResult:
    from yuni_agent.tools import builtin

    runtime = context.runtime
    auth = getattr(runtime, "image_auth", None)
    if auth is None or not auth.ready:
        raise ToolExecutionError("生图功能未配置或已关闭")
    prompt = arguments.prompt.strip()
    size = auth.default_size if arguments.aspect_ratio == "default" else ASPECT_SIZES[arguments.aspect_ratio]
    generation = runtime.data_generation

    reference: tuple[bytes, str] | None = None
    note = ""
    if arguments.mode == "companion_photo":
        reference = await asyncio.to_thread(_companion_avatar_reference, runtime)
        if reference is None:
            note = "（没有可用的头像参考图，本次按描述生成，长相可能与头像不同。）"

    if reference is not None:
        head = "以参考图中的人物为同一人，保持其长相、发型与气质一致，生成 TA 的自拍或日常生活照。"
        full_prompt = f"{head}{auth.style_prefix}，{prompt}" if auth.style_prefix else f"{head}{prompt}"
    else:
        full_prompt = f"{auth.style_prefix}，{prompt}" if auth.style_prefix else prompt

    try:
        if reference is not None:
            content = await builtin.edit_image(
                base_url=auth.url, api_key=auth.apikey.get_secret_value(), model=auth.model,
                prompt=full_prompt, image_bytes=reference[0], image_type=reference[1], size=size,
            )
        else:
            content = await builtin.request_image(
                base_url=auth.url, api_key=auth.apikey.get_secret_value(),
                model=auth.model, prompt=full_prompt, size=size,
            )
        image = await asyncio.to_thread(store_image, runtime.config.data_dir, content, prompt)
    except ImageGenerationError as exc:
        raise ToolExecutionError(str(exc)) from None
    if generation != runtime.data_generation:
        raise ToolExecutionError("数据已重置，图片未发送")
    return ToolResult(
        content=json.dumps({
            "status": "generated",
            "mode": arguments.mode,
            "usedAvatarReference": reference is not None,
            "description": prompt,
            "message": "图片已生成并暂存，会随你的下一次 reply 一起发给用户。请在 reply_reference 里说明画了什么，不要再次生图，也不要声称还在画。" + note,
        }, ensure_ascii=False),
        details={"image": image},
    )


IMAGE_TOOLS: list[RegisteredTool] = [
    RegisteredTool(
        name="generate_image",
        description=GENERATE_IMAGE_DESCRIPTION,
        arguments=GenerateImageArguments,
        risk=ToolRisk.WRITE,
        prepare=_prepare_generate_image,
        run=_generate_image,
        blocking=False,
        timeout=60.0,
        max_calls_per_turn=1,
        invalid_message="生图参数无效，请提供画面描述和合法画幅",
        enabled=lambda runtime: getattr(runtime, "image_generation_ready", False),
        group="image",
        summary="生成伴侣自拍日常照或按要求绘画创作图片",
        not_for="伴侣发送日常现成表情包由系统自动处理，无需调用此工具",
        requires_prior_reply=True,
        requires_followup_reply=True,
        redacted_fields=("prompt",),
    ),
]
