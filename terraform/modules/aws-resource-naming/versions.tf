# Tested working on Terraform 0.13.7 on 2021/--/--
terraform {
  required_providers {
    random = {
      source = "hashicorp/random"
    }
  }
  required_version = ">= 0.13"
}
