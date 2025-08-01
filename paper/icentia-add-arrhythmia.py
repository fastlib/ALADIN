import os
from tqdm import tqdm
import gzip
import pickle
import wfdb
import re
import json
import datetime
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


def find_patients(base_path):
    patients = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith("p") and len(dir_name) == 6:  # e.g., p00001
                patients.append({"patient": dir_name, "path": os.path.join(root, dir_name)})
    return patients

def get_results_from_folder(folder_path):

    ds = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl.gz'):

            relative_path = os.path.relpath(folder_path, os.environ.get('benchmark_data')+'/ICENTIA-results/ICENTIA-Dataset301')
            recordname = file_name[:-7]

            with gzip.open(os.path.join(folder_path, file_name), 'rb') as f:
                data = pickle.load(f)
                diagnoses = data.get("diagnosis", [])
                d = []
                for diagnosis in diagnoses:
                    onset = diagnosis.get("onset")
                    offset = diagnosis.get("offset")
                    duration = (offset - onset) / 250  # Convert to seconds

                    if diagnosis["type"] == "AFIB" and duration > 30:
                        d.append("AFIB")
                    elif diagnosis["type"] == "VT":
                        d.append("VT")
                    elif diagnosis["type"] == "SVT" and duration > 30:
                        d.append("SVT")
                    elif diagnosis["type"] == "TRIGEMINY":
                        d.append("TRIGEMINY")
                    elif diagnosis["type"] == "BIGEMINY":
                        d.append("BIGEMINY")
                    elif diagnosis["type"] == "IVR":
                        d.append("IVR")
                    elif diagnosis["type"] == "WENCKEBACH" or \
                            diagnosis["type"] == "AVB" or \
                            diagnosis["type"] == "AVB_TYPE2" or \
                            diagnosis["type"] == "SUDDEN_BRADY" or \
                            diagnosis["type"] == "IVB":
                        d.append(diagnosis["type"])
            

            ds.append({"record": relative_path+"/"+recordname, "predicted": d})#, "raw": diagnoses})
            
    return ds

def create_results_json(patients):

    rows = []
    for patient in tqdm(patients, desc="Processing patients"):
        patient_path = patient["path"]
        patient_results = get_results_from_folder(patient_path)
        for result in patient_results:
            row = {
                "record_id": result["record"],
                "diagnosis": list(set(result["predicted"])),  # Unique diagnoses
            }
            rows.append(row)

    print(rows)  # Print first 10 rows for debugging

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (os.environ["benchmark_data"])+"/aladin-466917-e056430d6165.json"
    creds = service_account.Credentials.from_service_account_file( (os.environ["benchmark_data"])+"/aladin-466917-e056430d6165.json")
    print(creds.project_id)
    client = bigquery.Client(credentials=creds, project=creds.project_id)
    JSONL_PATH = "icentia_results.jsonl"
    table_id = creds.project_id + ".benchmarks.ICENTIA-Staging"

    with open(JSONL_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

        print(f"Wrote {len(rows)} rows to {JSONL_PATH}")

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=[
            bigquery.SchemaField("record_id", "STRING"),
            bigquery.SchemaField("diagnosis", "JSON", mode="NULLABLE"),
        ],
        write_disposition="WRITE_TRUNCATE"
    )

    with open(JSONL_PATH, "rb") as source_file:
        load_job = client.load_table_from_file(
            source_file,
            table_id,
            job_config=job_config,
        )

    load_job.result()  # Wait for job to complete
    print(f"✅ Loaded {load_job.output_rows} rows into {table_id}.")

    # for rec in recs:
    #     row_to_insert = {
    #         "patient": rec[0][:-1],
    #         "record_id": rec[1],
    #         "status": "unprocessed",
    #         "last_updated": datetime.datetime.utcnow().isoformat()
    #     }
    #     rows_to_insert.append(row_to_insert)

    # with open(JSONL_PATH, "w") as f:
    #     for row in rows_to_insert:
    #         f.write(json.dumps(row) + "\n")

    #     print(f"Wrote {len(rows_to_insert)} rows to {JSONL_PATH}")

    # job_config = bigquery.LoadJobConfig(
    #     source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    #     schema=schema,
    #     write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    # )

    # with open(JSONL_PATH, "rb") as source_file:
    #     load_job = client.load_table_from_file(
    #         source_file,
    #         table_id,
    #         job_config=job_config,
    #     )



    # results["results"][0]["results"].sort(key=lambda x: x["record"])

    # with open(os.path.join(os.environ.get('benchmark_results'), "diagnosis", "set_level_diagnosis_ALADIN_ICENTIA_[" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "].json"), 'w') as f:
    #     json.dump(results, f, indent=4)


if __name__ == "__main__":
    basefolder = os.environ.get('benchmark_data')
    patients = find_patients(basefolder+'/ICENTIA-results/ICENTIA-Dataset301')
    print(f"Found {len(patients)} patient folders.")

    create_results_json(patients)
