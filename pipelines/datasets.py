"""Load and split UCI MAGIC/MiniBooNE classification data without scaler leakage.

Requires numpy and scikit-learn. Downloads are cached outside the repository.
Sources (CC BY 4.0):
    MAGIC: https://doi.org/10.24432/C52C8B (Bock, 2004)
    MiniBooNE: https://doi.org/10.24432/C5QC87 (Roe, 2005)
"""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


_DATASETS = {
    "magic": (
        "https://archive.ics.uci.edu/static/public/159/magic%2Bgamma%2Btelescope.zip",
        "magic04.data",
    ),
    "miniboone": (
        "https://archive.ics.uci.edu/static/public/199/miniboone%2Bparticle%2Bidentification.zip",
        "MiniBooNE_PID.txt",
    ),
}
_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
}


def _cached_file(dataset: str, cache_dir: Path) -> Path:
    """Download just the dataset member, writing the cache atomically."""
    url, filename = _DATASETS[dataset]
    destination = cache_dir / filename
    if destination.is_file():
        return destination
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response, NamedTemporaryFile() as archive:
        while chunk := response.read(1024 * 1024):
            archive.write(chunk)
        archive.flush()
        with ZipFile(archive.name) as zipped:
            members = [n for n in zipped.namelist() if Path(n).name == filename]
            if len(members) != 1:
                raise ValueError(f"Expected one {filename!r} in the UCI archive.")
            temporary_path = None
            try:
                with zipped.open(members[0]) as source, NamedTemporaryFile(
                    dir=cache_dir, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    while chunk := source.read(1024 * 1024):
                        temporary.write(chunk)
                temporary_path.replace(destination)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    return destination


def _read_dataset(dataset: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "magic":
        rows = np.loadtxt(path, delimiter=",", dtype=str, ndmin=2)
        if rows.shape[1] != 11 or not np.isin(rows[:, -1], ["g", "h"]).all():
            raise ValueError("MAGIC requires 10 numeric features and a g/h label.")
        X = rows[:, :-1].astype(np.float64)
        y = (rows[:, -1] == "g").astype(np.int64)
    else:
        with path.open() as source:
            counts = source.readline().split()
            if len(counts) != 2:
                raise ValueError("MiniBooNE requires a signal/background count header.")
            n_signal, n_background = map(int, counts)
            X = np.loadtxt(source, dtype=np.float64, ndmin=2)
        if min(n_signal, n_background) < 0 or X.shape != (
            n_signal + n_background, 50
        ):
            raise ValueError("MiniBooNE header counts or 50-feature rows are invalid.")
        # The original file has no label column: signal rows precede background.
        y = np.concatenate(
            (np.ones(n_signal, dtype=np.int64), np.zeros(n_background, dtype=np.int64))
        )
    if not np.isfinite(X).all():
        raise ValueError("Dataset features must contain only finite numeric values.")
    return X, y


def load_dataset(
    dataset: str = "magic",
    *,
    train_size: float | None = None,
    test_size: float | None = 0.2,
    preprocessing="standard",
    random_state: int | None = 42,
    shuffle: bool = True,
    stratify: bool = True,
    preprocess_test: bool = True,
    cache_dir: str | Path | None = None,
    data_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object | None]:
    """Return ``X_train, X_test, y_train, y_test, scaler``.

    Parameters
    ----------
    dataset : {"magic", "miniboone"}
        Case-insensitive dataset name. Targets are 1=signal, 0=background.
    train_size, test_size : float or None
        Fractions of the *full* dataset, each strictly between 0 and 1.
        If one is None, it complements the other. If both are None, use
        an 80/20 split. Explicit fractions may sum to less than 1; unused
        rows are omitted. scikit-learn rounds train down and test up.
    preprocessing : str, transformer, or None
        "standard" (default), "minmax", "robust", "maxabs", or "none"/None.
        A custom scikit-learn-compatible transformer is cloned before fitting.
        The scaler is fitted ONLY on X_train, never on X_test or unused rows.
        No preprocessing returns raw arrays and scaler=None.
    random_state : int or None
        Seed for reproducible partitions; None gives a fresh random split.
    shuffle, stratify : bool
        Shuffle and preserve class proportions by default. For ordered splits,
        set shuffle=False and stratify=False. MiniBooNE is ordered by class.
    preprocess_test : bool
        True transforms X_test with the fitted training scaler. False returns
        raw X_test so you can later call scaler.transform(X_test) yourself.
        X_train is transformed whenever preprocessing is enabled.
    cache_dir : path or None
        Download cache; defaults to ~/.cache/ml_sandbox/datasets.
    data_path : path or None
        Read a local original-format .data/.txt file instead of downloading.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test, scaler = load_dataset(
    ...     "magic", train_size=0.7, test_size=0.3
    ... )
    >>> X_train, X_test, y_train, y_test, scaler = load_dataset(
    ...     "miniboone", preprocessing="robust", preprocess_test=False
    ... )
    >>> X_test = scaler.transform(X_test)
    """
    if not isinstance(dataset, str) or dataset.lower() not in _DATASETS:
        raise ValueError("dataset must be 'magic' or 'miniboone'.")
    dataset = dataset.lower()
    for name, size in (("train_size", train_size), ("test_size", test_size)):
        if size is not None and (
            isinstance(size, (bool, np.bool_)) or not isinstance(size, Real)
            or not 0 < size < 1
        ):
            raise ValueError(f"{name} must be a float fraction between 0 and 1, or None.")
    if train_size is not None and test_size is not None and train_size + test_size > 1:
        raise ValueError("train_size + test_size must be at most 1.")
    if train_size is None and test_size is None:
        test_size = 0.2
    if stratify and not shuffle:
        raise ValueError("Set stratify=False when shuffle=False.")

    if preprocessing is None:
        scaler = None
    elif isinstance(preprocessing, str):
        style = preprocessing.lower()
        if style == "none":
            scaler = None
        elif style in _SCALERS:
            scaler = _SCALERS[style]()
        else:
            raise ValueError(f"Unknown preprocessing style: {preprocessing!r}.")
    else:
        scaler = clone(preprocessing)

    path = Path(data_path).expanduser() if data_path is not None else _cached_file(
        dataset,
        Path(cache_dir).expanduser() if cache_dir is not None else
        Path.home() / ".cache" / "ml_sandbox" / "datasets",
    )
    X, y = _read_dataset(dataset, path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y if stratify else None,
    )
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        if preprocess_test:
            X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler
