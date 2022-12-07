import os
import boto3

import dotenv
import logging

logger = logging.getLogger(__name__)


class S3Downloader:
    dotenv.load_dotenv()

    def __init__(self):
        self._access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self._secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self._service_name = os.getenv("SERVICE_NAME")
        self._region_name = os.getenv("REGION_NAME")
        self._bucket_name = os.getenv("BUCKET_NAME")

        self.s3 = boto3.resource(self._service_name,
                                 region_name=self._region_name,
                                 aws_access_key_id=self._access_key,
                                 aws_secret_access_key=self._secret_key,
                                 )

    def download_data_dir(self, prefix):
        bucket = self.s3.Bucket(self._bucket_name)
        logger.info(f"Downloading data from {bucket.name} bucket")

        for obj in bucket.objects.filter(Prefix=prefix):
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key)  # save to same path

        logger.info("***** Downloading data is done *****")
        logger.info(f"Dataset is saved in {prefix}")
