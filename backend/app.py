# Daniel Alvarez
# 8/16/26
# fastapi entry point

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import input_handler as ih

app = FastAPI(title="NeuroProfile")

CHUNK = 1 << 20


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/submit")
async def submit(url: str | None = Form(None),
                 text: str | None = Form(None),
                 file: UploadFile | None = File(None)):
    provided = [x for x in (url, text, file) if x]
    if len(provided) != 1:
        raise HTTPException(400, "provide exactly one of: url, text, file")

    try:
        if url:
            return ih.validate_url(url).dict()
        if text:
            return ih.validate_text(text).dict()

        # stream the upload to disk, stop early if it goes over the limit
        head = b""
        size = 0
        tmp = Path(tempfile.mkdtemp()) / (file.filename or "upload")
        with open(tmp, "wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if not head:
                    head = chunk[:16]
                if size > ih.MAX_UPLOAD_BYTES:
                    out.close(); tmp.unlink(missing_ok=True)
                    raise ih.InputError(f"file too large: exceeds {ih.MAX_UPLOAD_BYTES} bytes")
                out.write(chunk)
        return ih.validate_file(str(tmp), head, size).dict()

    except ih.InputError as e:
        raise HTTPException(400, str(e))