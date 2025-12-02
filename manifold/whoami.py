# manifold/whoami.py
from one.api import ONE

EID = "ebce500b-c530-47de-8cb1-963c552703ea"  # your current session

def _name(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("name", "nickname", "lab", "label"):
            if isinstance(x.get(k), str):
                return x[k]
    return str(x)

one = ONE(base_url="https://openalyx.internationalbrainlab.org")

# --- Session details ---
details = one.alyx.rest("sessions", "read", id=EID)
subject   = _name(details.get("subject"))
starttime = details.get("start_time") or details.get("date") or details.get("start")
lab       = _name(details.get("lab"))
projects  = details.get("projects", [])
if not isinstance(projects, (list, tuple)):
    projects = [projects]
projects = [_name(p) for p in projects]

print("=== SESSION DETAILS ===")
print("EID:     ", EID)
print("Subject: ", subject)
print("Start:   ", starttime)
print("Lab:     ", lab)
print("Projects:", projects)
print("Alyx URL:", details.get("url"))

# Cache path (where files are stored locally)
try:
    cache_path = one.eid2path(EID)
except Exception as e:
    cache_path = f"<unavailable> ({e})"
print("\nCache path:", cache_path)

# --- Objects & collections present ---
print("\n=== OBJECTS / COLLECTIONS ===")
objs = one.list(EID) or {}
for coll in sorted(objs):
    print(f"- {coll}: {sorted(objs[coll])}")

# --- Spikes datasets in likely collections ---
print("\n=== SPIKES DATASETS ===")
for coll in sorted([c for c in objs if ('probe' in c) or (c == 'alf')]):
    try:
        ds = one.list_datasets(EID, collection=coll, object="spikes")
        if ds:
            print(f"[{coll}]")
            for d in ds:
                if isinstance(d, dict):
                    print("  ", d.get("dataset_type") or d.get("rel_path") or d)
                else:
                    print("  ", d)
    except Exception as e:
        print(f"[{coll}] (error listing datasets: {e})")

# --- Resolve a spikes collection and show array shapes / files (if available) ---
for coll_try in ("alf/probe00/pykilosort", "alf/probe00", "alf", None):
    try:
        obj = one.load_object(EID, "spikes", collection=coll_try, attribute=["times", "clusters"])
        print("\nResolved spikes collection:", coll_try)
        t = obj.get("times"); c = obj.get("clusters")
        print("  times shape:   ", None if t is None else t.shape)
        print("  clusters shape:", None if c is None else c.shape)
        try:
            # .files() may not exist on some ONE versions
            print("  times file:   ", obj.files("times"))
            print("  clusters file:", obj.files("clusters"))
        except Exception:
            print("  (.files accessor not available on this ONE version)")
        break
    except Exception:
        continue
