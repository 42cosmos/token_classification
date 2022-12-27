#!/usr/bin/env bash
mkdir token_data
aws s3 sync s3://{my_bucket_name}/{my_object_name} ./token_data --no-sign-request

unzip ./token_data/klue/klue.zip -d ./token_data/klue/
rm -rf ./token_data/klue/klue.zip

python3 main.py \
	--do_train \
	--do_eval \
	--dset_name klue

python3 main.py \
	--do_train \
	--do_eval \
	--load_checkpoint \
	--dset_name docent