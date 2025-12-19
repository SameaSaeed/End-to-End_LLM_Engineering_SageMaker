##### **Build and push Docker image to ECR**



###### Create an ECR repository:

aws ecr create-repository --repository-name vllm-llama



###### 

###### Build Docker image:

docker build -t vllm-llama ./vllm\_container





###### Tag the image for ECR:

docker tag vllm-llama:latest <account\_id>.dkr.ecr.<region>.amazonaws.com/vllm-llama:latest





###### Login to ECR and push:

aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account\_id>.dkr.ecr.<region>.amazonaws.com

docker push <account\_id>.dkr.ecr.<region>.amazonaws.com/vllm-llama:latest

