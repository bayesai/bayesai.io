bucket = "bayesai-terraform-state"
key = "bayesai.io-prod-terraform.tfstate"
# encrypt on the backend
encrypt = true
# use state locking
dynamodb_table = "dynamo-preprod-terraform-state-locks"
region = "us-east-1"
