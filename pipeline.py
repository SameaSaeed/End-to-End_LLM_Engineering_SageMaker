import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import (
    Compute,
    SourceCode,
    InputData,
    OutputDataConfig,
    CheckpointConfig,
    StoppingCondition,
)
from sagemaker.modules.distributed import Torchrun
from sagemaker.workflow.model_step import ModelStep
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import image_uris

# -------------------------------------------------------------------
# Session / globals
# -------------------------------------------------------------------
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sess.default_bucket()
prefix = sess.default_bucket_prefix or ""

model_id = "meta-llama/Llama-3.1-8B-Instruct"
job_name = f"train-{model_id.split('/')[-1].replace('.', '-')}-sft"

input_path = f"s3://{bucket}/{prefix}/datasets/llm-fine-tuning"
output_path = f"s3://{bucket}/{prefix}/{job_name}"

# -------------------------------------------------------------------
# Images
# -------------------------------------------------------------------
train_image = image_uris.retrieve(
    framework="huggingface",
    region=sess.boto_session.region_name,
    version="4.49.0",
    base_framework_version="pytorch2.5.1",
    instance_type="ml.p4d.24xlarge",
    image_scope="training",
)

infer_image = image_uris.retrieve(
    framework="huggingface",
    region=sess.boto_session.region_name,
    version="4.49.0",
    instance_type="ml.g5.12xlarge",
    image_scope="inference",
)

# -------------------------------
# 1️⃣ DATA PREPROCESSING STEP
# -------------------------------
prep_step = ProcessingStep(
    name="PrepareDataset",
    processor=ScriptProcessor(
        image_uri=train_image,
        command=["python3"],
        instance_type="ml.m5.4xlarge",
        instance_count=1,
        role=role,
    ),
    code="./scripts/data_preprocess.py",
    outputs=[
        ProcessingOutput(source="/opt/ml/output/train", destination=f"{input_path}/train"),
        ProcessingOutput(source="/opt/ml/output/val", destination=f"{input_path}/val"),
    ],
)

# -------------------------------
# 2️⃣ TRAINING STEP
# -------------------------------
trainer = ModelTrainer(
    training_image=train_image,
    source_code=SourceCode(
        source_dir="./scripts",
        entry_script="train.py",
        requirements="requirements.txt",
    ),
    base_job_name=job_name,
    compute=Compute(instance_type="ml.p4d.24xlarge", instance_count=1),
    distributed=Torchrun(),
    stopping_condition=StoppingCondition(max_runtime_in_seconds=18000),
    hyperparameters={"config": "/opt/ml/input/data/config/args.yaml"},
    output_data_config=OutputDataConfig(s3_output_path=output_path),
    checkpoint_config=CheckpointConfig(
        s3_uri=f"{output_path}/checkpoints", local_path="/opt/ml/checkpoints"
    ),
)

train_step = trainer.to_pipeline_step(
    name="TrainModel",
    input_data_config=[
        InputData("train", f"{input_path}/train"),
        InputData("val", f"{input_path}/val"),
        InputData("config", f"{input_path}/config"),
    ],
    depends_on=[prep_step],
)

# -------------------------------
# 3️⃣ LOAD LAST MODEL STEP
# -------------------------------
load_model_step = ProcessingStep(
    name="LoadLastModel",
    processor=ScriptProcessor(
        image_uri=infer_image,
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
    ),
    code="./scripts/load.py",  # your script to load previous model
    inputs=[
        ProcessingInput(source=f"{output_path}/checkpoints", destination="/opt/ml/input/checkpoints")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/output/loaded_model", destination=f"{output_path}/loaded_model")
    ],
    depends_on=[train_step],  # run after training
)

# -------------------------------
# 4️⃣ DEPLOYMENT STEP
# -------------------------------
deploy_model = HuggingFaceModel(
    model_data=load_model_step.outputs[0].destination,  # use loaded model
    image_uri=infer_image,
    role=role,
    env={"HF_MODEL_ID": "/opt/ml/model", "SM_NUM_GPUS": "1", "MESSAGES_API_ENABLED": "true"},
)

deploy_step = ModelStep(
    name="DeployModel",
    model=deploy_model,
    inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.g5.12xlarge", initial_instance_count=1),
)

# -------------------------------
# 5️⃣ EVALUATION STEP
# -------------------------------
eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=ScriptProcessor(
        image_uri=infer_image,
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
    ),
    code="./scripts/evaluate.py",
    inputs=[ProcessingInput(source=f"{input_path}/test", destination="/opt/ml/input/data")],
    outputs=[ProcessingOutput(source="/opt/ml/output", destination=f"{output_path}/evaluation")],
    depends_on=[deploy_step],
)

# -------------------------------
# 6️⃣ TESTING STEP
# -------------------------------
giskard_step = ProcessingStep(
    name="RunGiskardTests",
    processor=ScriptProcessor(
        image_uri=infer_image,
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
    ),
    code="./test/giskard_test.py",
    inputs=[ProcessingInput(source=f"{input_path}/test", destination="/opt/ml/input/data")],
    outputs=[ProcessingOutput(source="/opt/ml/output", destination=f"{output_path}/test_results")],
    depends_on=[eval_step],
)

# -------------------------------
# PIPELINE
# -------------------------------
pipeline = Pipeline(
    name="LLM-SFT-ModelTrainer-Pipeline",
    steps=[prep_step, train_step, load_model_step, deploy_step, eval_step, giskard_step],
)

pipeline.upsert(role_arn=role)
pipeline.start()
