import re
import urllib.request
from pathlib import Path

CLOUDFRONT_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def month_range(start: str, end: str):
    def parse(s):
        if not re.match(r"\d{4}-\d{2}", s):
            raise ValueError("Formato YYYY-MM")
        y, m = map(int, s.split("-"))
        return y, m

    sy, sm = parse(start)
    ey, em = parse(end)

    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def download(service: str, start: str, end: str, raw_dir: Path):
    months = month_range(start, end)
    for mm in months:
        year = mm[:4]
        fname = f"{service}_tripdata_{mm}.parquet"
        url = f"{CLOUDFRONT_BASE}/{fname}"
        out = raw_dir / "tlc" / service / year / fname
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.exists():
            print(f"SKIP {fname}")
            continue

        print(f"DESCARGANDO {fname}")
        urllib.request.urlretrieve(url, out)


# Tal vez debería hacerse en otro archivo : download_lookup?
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

def ensure_taxi_zone_lookup(lookup_path: Path, overwrite: bool = False) -> str:
    """
    Descarga taxi_zone_lookup.csv si no existe (o si overwrite=True).
    """
    lookup_path.parent.mkdir(parents=True, exist_ok=True)

    if lookup_path.exists() and not overwrite:
        return f"SKIP lookup ({lookup_path.name})"

    urllib.request.urlretrieve(LOOKUP_URL, lookup_path)
    return f"OK   lookup ({lookup_path.name})"