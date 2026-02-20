import time, pickle, zlib
from multiprocessing import Manager
from src.globals import ROUND_TIMEOUT_S
 
# Initialize Manager lazily to avoid Ray conflicts
_mgr = None
COMM = None

def _get_comm():
    global _mgr, COMM
    if _mgr is None:
        _mgr = Manager()
        COMM = _mgr.dict()
    return COMM
 
def write_blob(cid_from: int, rnd: int, sub: int, payload: dict, log_func):
    comm = _get_comm()
    blob = zlib.compress(pickle.dumps(payload), 6)
    comm[(cid_from, rnd, sub)] = blob
    kb = len(blob) / 1024
    log_func(f"[Comm] wrote {kb:.1f} KB key={(cid_from, rnd, sub)}")
    return kb
 
def read_blob(cid_from: int, rnd: int, log_func):
    comm = _get_comm()
    for sub in range(3):
        key = (cid_from, rnd, sub)
        for attempt in range(3):
            if key in comm:
                blob = comm[key]
                purge(cid_from, rnd)
                log_func(f"[Comm] read key={key} OK")
                return pickle.loads(zlib.decompress(blob))
            time.sleep(10)  # Increased from 5 to 10 seconds for better teacher-student coordination
            log_func(f"[Comm] Attempt {attempt+1}/3 failed for key={key}, waiting longer...")
    log_func(f"[Comm] Timeout reading from CID{cid_from} round{rnd}")
    raise TimeoutError
 
def purge(cid_from: int, rnd: int):
    comm = _get_comm()
    for k in list(comm.keys()):
        if k[:2] == (cid_from, rnd):
            comm.pop(k, None)