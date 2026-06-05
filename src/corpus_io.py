"""
Manifest do corpus + download dos documentos (apoio à Etapa 2).

O manifest é uma lista de dicts com, no mínimo:
    id, titulo, fonte, categoria, subcategoria, ano, vigencia, tipo, url
Após o download, cada item ganha: local_path, downloaded (bool), bytes, error.
"""
from __future__ import annotations

import json
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}


def load_manifest(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(manifest: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _ext_for(item: dict, resp) -> str:
    tipo = (item.get("tipo") or "").lower()
    if tipo in {"pdf", "html", "htm", "docx", "txt"}:
        return ".htm" if tipo == "htm" else f".{tipo}"
    ct = (resp.headers.get("Content-Type", "") if resp is not None else "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "html" in ct:
        return ".html"
    if "word" in ct or "officedocument" in ct:
        return ".docx"
    url = item.get("url", "").lower()
    for e in (".pdf", ".docx", ".html", ".htm", ".txt"):
        if url.split("?")[0].endswith(e):
            return e
    return ".pdf"


def download_one(item: dict, raw_dir: str | Path, *, timeout: int = 90, overwrite: bool = False) -> dict:
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # alguns hosts (ex.: CRESESB) têm certificado TLS com nome incorreto: o item
    # pode marcar "insecure": true para baixar ignorando a verificação de cert.
    verify = not item.get("insecure", False)
    if not verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        with requests.get(item["url"], headers=HEADERS, stream=True, timeout=timeout,
                          allow_redirects=True, verify=verify) as r:
            r.raise_for_status()
            ext = _ext_for(item, r)
            dest = raw_dir / f"{item['id']}{ext}"
            if dest.exists() and not overwrite:
                item.update(local_path=str(dest), downloaded=True, bytes=dest.stat().st_size, error="")
                return item
            size = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)
            item.update(local_path=str(dest), downloaded=size > 0, bytes=size, error="")
    except Exception as e:  # noqa: BLE001
        item.update(local_path="", downloaded=False, bytes=0, error=f"{type(e).__name__}: {e}")
    return item


def download_manifest(manifest: list[dict], raw_dir: str | Path, *, overwrite: bool = False,
                      timeout: int = 120) -> list[dict]:
    for item in manifest:
        download_one(item, raw_dir, overwrite=overwrite, timeout=timeout)
        flag = "OK " if item.get("downloaded") else "FALHA"
        kb = item.get("bytes", 0) / 1024
        print(f"[{flag}] {item['id']:<28} {kb:8.1f} KB  {item.get('error','')}")
    ok = sum(1 for i in manifest if i.get("downloaded"))
    print(f"\n{ok}/{len(manifest)} documentos baixados.")
    return manifest
