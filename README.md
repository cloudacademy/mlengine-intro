# Introduction to Google Cloud Machine Learning Engine
This file contains text you can copy and paste for the examples in Cloud Academy's _Introduction to Google Cloud Machine Learning Engine_ course.  

### TensorFlow
TensorFlow website: https://www.tensorflow.org  
TensorFlow installation: https://www.tensorflow.org/install  

```
python -V       # Check which version of Python 2 is installed
python3 -V      # Check which version of Python 3 is installed
pip install --user --upgrade pip
pip install --user --upgrade virtualenv
virtualenv mlenv
source mlenv/bin/activate
pip install tensorflow==1.10
pip install pandas
```

```
git clone https://github.com/cloudacademy/mlengine-intro.git
cd mlengine-intro/iris/trainer
python iris.py
```

### Training a Model with ML Engine
Google Cloud SDK installation: https://cloud.google.com/sdk  

```
cd ..
gcloud ai-platform local train --module-name trainer.iris --package-path trainer
```

```
BUCKET=gs://[ProjectID]-ml  # Replace [ProjectID] with your Google Cloud Project ID  
REGION=[Region]  # Replace [Region] with a Google Cloud Platform region, such as us-central1  
```
```
gcloud ai-platform jobs submit training iris1 \
    --module-name trainer.iris \
    --package-path trainer \
    --staging-bucket $BUCKET \
    --region $REGION \
    --runtime-version 1.10
```

### Feature Engineering
Google's original sample code: https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census

```
cd ../census/estimator
```

```
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer \
    -- \
    --train-files data/adult.data.csv \
    --eval-files data/adult.test.csv \
    --model-type wide
```

### A Wide and Deep Model
```
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer \
    -- \
    --train-files data/adult.data.csv \
    --eval-files data/adult.test.csv \
    --model-type deep
```

### Distributed Training on ML Engine
Hyperparameter Tuning: https://cloud.google.com/ml-engine/docs/concepts/hyperparameter-tuning-overview  

```
gsutil cp -r gs://cloudml-public/census/data $BUCKET  
TRAIN_DATA=$BUCKET/data/adult.data.csv  
EVAL_DATA=$BUCKET/data/adult.test.csv  
JOB=census1  
```

```
gcloud ai-platform jobs submit training $JOB \
    --job-dir $BUCKET/$JOB \
    --runtime-version 1.10 \
    --module-name trainer.task \
    --package-path trainer \
    --region $REGION \
    --scale-tier STANDARD_1 \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA
```

### Deploying a Model on ML Engine
```
gcloud ai-platform models create census --regions=$REGION  
gsutil ls -r $BUCKET/census1/export  
```
```
# Note: Replace [Path-to-model] below with your Cloud Storage path
gcloud ai-platform versions create v1 \
    --model census \
    --runtime-version 1.10 \
    --origin [Path-to-model]
```
```
gcloud ai-platform predict \
    --model census \
    --version v1 \
    --json-instances \
    ../test.json
```

### Conclusion
Cloud Machine Learning Engine documentation: https://cloud.google.com/ai-platform/docs  
