# Bayesai.io site

This directory contains the code to setup the infrastructure for the site

![Architecture](./docs/site.png)

This will setup a code build pipeline for production

We currently have 1 environment: prod

Below is the code to apply and update the terraform. Please remember to set your AWS credentials and change preprod.tfvars to the correct variables file for your environment.

```
# This only needs to be run ONCE per AWS ACCOUNT to setup remote state

cd remote-state
terraform init
terraform apply

# This only needs to be run on setting up a codebuild pipeline or when changing environments
terraform init -backend-config=prod-backend.tfvars

terraform plan --var-file ./prod.tfvars
terraform apply --var-file ./prod.tfvars
```
