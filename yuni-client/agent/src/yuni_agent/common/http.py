from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException


def fail(status_code: int, code: int, message: str) -> NoReturn:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})
