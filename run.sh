#!/usr/bin/env bash
mkdir token_data
aws s3 sync s3://{my_bucket_name}/{my_object_name} ./token_data --no-sign-request

python3 inference.py --dset_name klue
