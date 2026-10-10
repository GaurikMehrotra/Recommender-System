"""Deterministic synthetic MovieLens-shaped data for offline functional validation."""
import argparse
from pathlib import Path
import pandas as pd


def generate(path: str | Path, users: int = 12, items: int = 40, events_per_user: int = 10) -> Path:
    """Create fictional ratings/movie metadata; refuse overwriting existing files."""
    if events_per_user > items or events_per_user < 7:
        raise ValueError("Need at least 7 events per user and at most catalog size")
    path=Path(path)
    path.mkdir(parents=True,exist_ok=True)
    if (path / "u.data").exists() or (path / "u.item").exists():
        raise FileExistsError("Refusing to overwrite synthetic fixture")
    ratings=[[u,(u*3+t)%items+1,4+t%2,100+t] for u in range(1,users+1) for t in range(events_per_user)]
    movies=[[i,f"Synthetic {i} ({1980+i%25})","01-Jan-1995","","",*[int(g==i%19) for g in range(19)]] for i in range(1,items+1)]
    pd.DataFrame(ratings).to_csv(path / "u.data",sep="\t",header=False,index=False)
    pd.DataFrame(movies).to_csv(path / "u.item",sep="|",header=False,index=False)
    return path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="data/synthetic/ml-100k")
    args=parser.parse_args()
    print(generate(args.output))

if __name__ == "__main__":
    main()
