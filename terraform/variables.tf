variable "environment" {
  description = "AWS environment name"
  default     = ""
}

variable "project" {
  description = "project name"
  default     = ""
}

variable "owner" {
  description = "Contact point for this infrastructure"
  default     = ""
}

variable "allowed_account_ids" {
  description = "List of allowed AWS account ids where resources can be created"
  type = list(string)
  default = []
}

variable "s3_state_bucket" {
  description = "An AWS region to put resources in"
  default     = ""
}

variable "region" {
  description = "An AWS region to put resources in"
  default     = ""
}

variable "hosted_zone" {
  description = "Hosted route53 dns zone"
  default     = ""
}

variable "domain_name" {
  type        = string
  description = "The domain name for the website."
}
