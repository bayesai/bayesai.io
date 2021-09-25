bucket = "bayesai-terraform-state-bucket"
key = "bayesai.io-prod-terraform.tfstate"
# encrypt on the backend
encrypt = true
# use state locking
dynamodb_table = "dynamo-prod-terraform-state-locks"
region = "us-east-1"
