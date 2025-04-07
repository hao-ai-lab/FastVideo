import json
import os
import sys
import uuid

import requests

API_KEY = os.environ['RUNPOD_API_KEY']
RUN_ID = os.environ.get('GITHUB_RUN_ID', str(uuid.uuid4()))
PODS_API = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def get_job_ids():
    """Parse job IDs from environment variable"""
    job_ids_str = os.environ.get('JOB_IDS')
    try:
        job_ids = json.loads(job_ids_str)
        if not isinstance(job_ids, list):
            print("Error: JOB_IDS is not a list.")
            sys.exit(1)
        return job_ids
    except json.JSONDecodeError as e:
        print(f"Error parsing JOB_IDS: {e}")
        sys.exit(1)


def cleanup_pods():
    """Find and terminate RunPod instances"""
    print(f"Run ID: {RUN_ID}")

    single_job_id = os.environ.get('JOB_ID')
    
    if single_job_id:
        job_ids = [single_job_id]
        print(f"Job ID: {single_job_id}")
    else:
        job_ids = get_job_ids()
        print(f"Job IDs: {job_ids}")

    # Get all pods associated with RunPod API_KEY
    try:
        response = requests.get(PODS_API, headers=HEADERS)
        response.raise_for_status()
        pods = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting pods: {e}")
        sys.exit(1)

    # Find and terminate pods created by this workflow run
    terminated_pods = []
    for pod in pods:
        pod_name = pod.get("name", "")
        pod_id = pod.get("id")

        # Check if this pod was created by one of our jobs
        if any(f"{job_id}-{RUN_ID}" in pod_name for job_id in job_ids):
            print(f"Found pod: {pod_id} ({pod_name})")
            try:
                print(f"Terminating pod {pod_id}...")
                term_response = requests.delete(f"{PODS_API}/{pod_id}",
                                                headers=HEADERS)
                term_response.raise_for_status()
                terminated_pods.append(pod_id)
                print(f"Successfully terminated pod {pod_id}")
            except requests.exceptions.RequestException as e:
                print(f"Error terminating pod {pod_id}: {e}")
                sys.exit(1)
    
    if terminated_pods:
        if single_job_id:
            print(f"Terminated pod: {terminated_pods[0]}")
        else:
            print(f"Terminated {len(terminated_pods)} pods: {terminated_pods}")
    else:
        if single_job_id:
            print(f"No pod found matching pattern: {single_job_id}-{RUN_ID}")
        else:
            print("No pods found to terminate.")


def main():
    cleanup_pods()


if __name__ == "__main__":
    main()
