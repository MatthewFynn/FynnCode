# csv_safe.py
import os, io, csv, shutil, tempfile

def atomic_write_bytes(path: str, data: bytes) -> None:
    """
    Crash-safe overwrite: write bytes to a temp file in the SAME directory,
    flush+fsync, then atomically replace the target.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=os.path.splitext(path)[1] or ".tmp")
    try:
        with os.fdopen(fd, "wb", closefd=True) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())             # ensure data is on disk
        os.replace(tmp_path, path)            # atomic swap (POSIX & Windows)
        # fsync the directory entry (best effort)
        try:
            dir_fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
    finally:
        if os.path.exists(tmp_path):          # cleanup if something exploded before replace
            try: os.remove(tmp_path)
            except Exception: pass

def rows_to_csv_text(rows, headers) -> str:
    """
    Convert a list of dicts (or lists/tuples) to CSV text with given headers.
    - If rows are dicts, values are taken in header order (missing -> empty).
    - If rows are sequences, they must match header length.
    """
    sio = io.StringIO(newline="")  # handle universal newlines correctly
    writer = csv.writer(sio)
    writer.writerow(headers)
    for r in rows:
        if isinstance(r, dict):
            writer.writerow([r.get(h, "") for h in headers])
        else:
            writer.writerow(list(r))
    return sio.getvalue()

def read_csv_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()

def validate_csv(path: str) -> bool:
    """
    Lightweight validator: ensures we can parse all rows with Python's csv reader.
    Also checks that all rows have the same number of columns as the header.
    """
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not header: return False
            n = len(header)
            for row in reader:
                if len(row) != n:
                    # allow trailing blank line
                    if len(row) == 0: 
                        continue
                    return False
        return True
    except Exception:
        return False

def update_master_csv(master_path: str, new_rows_csv_text: str, keep_backup: bool = True) -> None:
    """
    Append new CSV rows to master with ATOMIC REPLACE and optional rolling backup.
    - Preserves existing master header if present.
    - Removes duplicate header from new_rows_csv_text if it exists.
    """
    # Read existing (if any)
    old = ""
    if os.path.exists(master_path):
        old = read_csv_text(master_path)

    # Split into lines for header handling
    old_lines = [ln for ln in old.splitlines()]
    new_lines = [ln for ln in new_rows_csv_text.splitlines() if ln.strip() != ""]

    if not new_lines:
        return  # nothing to add

    # Determine headers
    new_header = new_lines[0]
    if old_lines:
        old_header = old_lines[0]
        # If headers match, drop the new header before appending
        if new_header.strip() == old_header.strip():
            new_body = new_lines[1:]
            combined_lines = old_lines + new_body
        else:
            # Different headers → start fresh with new header and append old body (optional policy)
            # Here we choose to keep the existing header and only append matching bodies.
            # If headers differ, we keep old header and append rows that match column count.
            old_cols = len(list(csv.reader([old_header]))[0])
            new_rows = list(csv.reader(new_lines))
            # keep rows (skip new header) with correct width
            filtered = [r for r in new_rows[1:] if len(r) == old_cols]
            sio = io.StringIO(newline="")
            w = csv.writer(sio)
            w.writerow(list(csv.reader([old_header]))[0])
            for r in csv.reader([old]):  # write old body quickly (including header) below
                pass  # we'll just reuse old text instead of rewriting
            # Since we already have old text, just re-assemble by text approach:
            combined_lines = old_lines + [",".join(r) for r in filtered]
    else:
        # No old file → keep the new content as-is
        combined_lines = new_lines

    # Ensure single trailing newline
    combined_text = "\n".join(combined_lines) + "\n"

    # Atomic write
    atomic_write_bytes(master_path, combined_text.encode("utf-8"))

    # Rolling backup
    if keep_backup:
        try:
            shutil.copy2(master_path, master_path + ".bak")
        except Exception:
            pass

def recover_master_if_corrupt(master_path: str) -> bool:
    """
    If master is unreadable/corrupt (per validate_csv), promote .bak to master.
    Returns True if a recovery was performed.
    """
    ok = validate_csv(master_path)
    if ok:
        return False
    bak = master_path + ".bak"
    if os.path.exists(bak):
        try:
            shutil.copy2(bak, master_path)
            return True
        except Exception:
            return False
    return False
