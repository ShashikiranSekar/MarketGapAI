import pandas as pd
from config.settings import AUDIT_CHUNK_SIZE, REQUIRED_DATASETS, get_data_dir

def audit_dataset_file(name):
    filename=REQUIRED_DATASETS[name]
    path=get_data_dir()/filename
    if not path.is_file():
        return {"dataset":name,"filename":filename,"status":"Missing","rows":None,"columns":None,"size_mb":None,"missing_cells":None,"duplicates_within_chunks":None,"column_details":{},"error":"File not found"}
    try:
        cols=pd.read_csv(path,nrows=0).columns.tolist()
        missing={c:0 for c in cols}; types={c:set() for c in cols}; rows=dupes=0
        for chunk in pd.read_csv(path,chunksize=AUDIT_CHUNK_SIZE,low_memory=False):
            rows+=len(chunk); dupes+=int(chunk.duplicated().sum())
            for c in cols:
                missing[c]+=int(chunk[c].isna().sum()); types[c].add(str(chunk[c].dtype))
        detail={c:{"dtype":", ".join(sorted(types[c])) or "unknown","missing":missing[c],"missing_percent":round(100*missing[c]/rows,2) if rows else 0} for c in cols}
        return {"dataset":name,"filename":filename,"status":"Passed","rows":rows,"columns":len(cols),"size_mb":round(path.stat().st_size/(1024*1024),2),"missing_cells":sum(missing.values()),"duplicates_within_chunks":dupes,"column_details":detail,"error":None}
    except Exception as exc:
        return {"dataset":name,"filename":filename,"status":"Error","rows":None,"columns":None,"size_mb":round(path.stat().st_size/(1024*1024),2),"missing_cells":None,"duplicates_within_chunks":None,"column_details":{},"error":str(exc)}

def audit_all_datasets():
    rows=[audit_dataset_file(k) for k in REQUIRED_DATASETS]
    return pd.DataFrame([{"Dataset":r["dataset"],"File":r["filename"],"Status":r["status"],"Rows":r["rows"],"Columns":r["columns"],"Size (MB)":r["size_mb"],"Missing Cells":r["missing_cells"],"Duplicate Rows (Within Chunks)":r["duplicates_within_chunks"],"Error":r["error"]} for r in rows])
