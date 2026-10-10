"""Download MovieLens directly from GroupLens; never redistribute the dataset."""

import argparse
import hashlib
import urllib.request
import zipfile
import zlib
from pathlib import Path

URLS = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}


def download(size: str, root: str | Path = "data/raw", expected_sha256: str | None = None) -> Path:
    """Fetch with TLS verification and optionally verify a user-supplied trusted digest.

    Record SHA256 for provenance; a freshly computed digest is not authenticity proof.
    The official HTTPS origin provides transport authentication.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"ml-{size}.zip"
    target = root / f"ml-{size}"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(URLS[size], temporary)
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("MovieLens checksum mismatch; remove the archive and investigate")
    archive.with_suffix(".zip.sha256").write_text(digest + "\n")
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive path")
            if destination.exists() and not member.is_dir():
                if zlib.crc32(destination.read_bytes()) & 0xFFFFFFFF != member.CRC:
                    raise ValueError("Existing data differs from archive; refusing to overwrite")
        corrupt = z.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt ZIP member: {corrupt}")
        for member in z.infolist():
            if not (root / member.filename).exists():
                z.extract(member, root)
    required = ["u.data", "u.item"] if size == "100k" else ["ratings.dat", "movies.dat"]
    if not all((target / name).is_file() for name in required):
        raise ValueError("Archive does not contain the expected MovieLens files")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=URLS, default="100k")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    print(download(args.size, args.root, args.sha256))


if __name__ == "__main__":
    main()
