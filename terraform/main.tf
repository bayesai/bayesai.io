terraform {
  required_version = ">= 0.13"
  backend "s3" {}
}

locals {
  common_tags = {
    Environment  = var.environment
    Project      = var.project
    Owner        = var.owner
    Created_By   = "terraform"
  }
  extra_tags  = {
  }
}

# ------------------------------------------------------------------------------
# CONFIGURE OUR AWS CONNECTION
# ------------------------------------------------------------------------------

provider "aws" {
  region = var.region
  allowed_account_ids = var.allowed_account_ids
}

# ------------------------------------------------------------------------------
# CREATE THE S3 BUCKET
# ------------------------------------------------------------------------------

resource "aws_s3_bucket" "terraform_state" {
  # TODO: change this to your own name! S3 bucket names must be *globally* unique.
  bucket = var.s3_state_bucket

  # Enable versioning so we can see the full revision history of our
  # state files
  versioning {
    enabled = true
  }

  # Enable server-side encryption by default
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = merge( local.common_tags, local.extra_tags)
}

# ------------------------------------------------------------------------------
# CREATE THE DYNAMODB TABLE
# ------------------------------------------------------------------------------

resource "aws_dynamodb_table" "terraform_locks" {
  name         = "dynamo-${var.environment}-terraform-state-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
  tags = merge( local.common_tags, local.extra_tags)
}

# ------------------------------------------------------------------------------
# CODE BUILD
# ------------------------------------------------------------------------------

module "terraform_ci_cd" {
  source            = "./modules/code-build"
  terraform_version = ">= 0.13"
  project           = var.project
  environment       = var.environment
  owner             = var.owner

  source_repository_url = "https://github.com/bayesai/bayesai.io"

  ci_env_var = [
    # {
    #   "name"  = "MY_SECRET"
    #   "value" = "MY_SECRET_VALUE"
    # },
    # {
    #   "name"  = "MY_SECRET_2"
    #   "value" = "MY_SECRET_VALUE_2"
    #   "type"  = "PARAMETER_STORE"
    # },
  ]

  ci_install_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  ci_pre_build_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  ci_build_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  ci_post_build_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  cd_env_var = [
    {
      "name"  = "MY_SECRET"
      "value" = "MY_SECRET_VALUE"
    },
    {
      "name"  = "MY_SECRET_2"
      "value" = "MY_SECRET_VALUE_2"
      "type"  = "PARAMETER_STORE"
    },
  ]

  cd_install_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  cd_pre_build_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  cd_build_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  cd_post_build_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]
}

# ------------------------------------------------------------------------------
# STATIC SITE
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "content" {
  bucket = "bayesai-io-prod-content-bucket"
}

data "aws_iam_policy_document" "content" {
  # Grant read to the module's OAI
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${module.external_bucket_static_site.content_bucket_arn}/*"]
    principals {
      type        = "AWS"
      identifiers = [module.external_bucket_static_site.cloudfront_oai_iam_arn]
    }
  }
}
resource "aws_s3_bucket_policy" "content" {
  bucket = aws_s3_bucket.content.id
  policy = data.aws_iam_policy_document.content.json
}

module "external_bucket_static_site" {
  source = "./modules/static-site"
  domain_name = "bayesai.io"

  acm_certificate_arn = var.acm_certificate_arn

  # Optional
  hosted_zone_id               = var.hosted_zone
  default_subdirectory_object  = "index.html"
  create_content_bucket        = false
  manage_content_bucket_policy = false
  content_bucket_name          = aws_s3_bucket.content.id
  force_destroy_buckets        = true
  tags                         = merge( local.common_tags, local.extra_tags)
}

