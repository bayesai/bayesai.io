terraform {
  required_providers {
    archive = {
      source = "hashicorp/archive"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.41"
    }
    cloudflare = {
      source = "registry.terraform.io/cloudflare/cloudflare"
    }
  }
  required_version = ">= 0.13"
}
