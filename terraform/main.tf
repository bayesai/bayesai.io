terraform {
  required_version = ">= 0.13"
  backend "s3" {
    # bucket = var.s3_state_bucket
    # key = var.key
    # region = var.region
    # dynamodb_table = "dynamo-${var.environment}-terraform-state-locks"
  }
  
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
# CODE BUILD
# ------------------------------------------------------------------------------

module "terraform_ci_cd" {
  source            = "./modules/code-build"
  terraform_version = ">= 0.13"
  project           = var.project
  environment       = var.environment
  owner             = var.owner
  enable_ci         = false

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

  ### These commands are run when a new PR is created ###
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

  ### These commands are run on a commit to master ###
  cd_env_var = [
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

  cd_install_commands = [
    "gem install jekyll bundler",
    "bundle",
  ]

  cd_pre_build_commands = [
    "echo 'custom command 1'",
    "echo 'custom command 2'",
  ]

  cd_build_commands = [
    "echo '******** Building Jekyll site ********'",
    "JEKYLL_ENV=production jekyll build",
    "echo '******** Uploading to S3 ********'",
    "aws s3 sync --delete _site/ s3://www.bayesai.io"
  ]

  cd_post_build_commands = [
    "echo Build completed on `date`",
  ]
}

# ------------------------------------------------------------------------------
# STATIC SITE
# ------------------------------------------------------------------------------

module "external_bucket_static_site" {
  source = "./modules/static-site"
  domain_name = var.domain_name

  # Optional
  index_redirect               = true
  default_subdirectory_object  = "index.html"
  create_content_bucket        = true
  manage_content_bucket_policy = true
  content_bucket_name          = "bayesai.io"
  force_destroy_buckets        = true
  tags                         = merge( local.common_tags, local.extra_tags)
}

