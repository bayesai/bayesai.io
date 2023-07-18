provider "cloudflare" {
  api_user_service_key = "ZZZZZZZ"
}

data "aws_canonical_user_id" "current_user" {}
locals {
  content_bucket_name = coalesce(var.content_bucket_name, "${var.domain_name}-static-content")
  content_bucket      = var.create_content_bucket ? aws_s3_bucket.content[0] : data.aws_s3_bucket.content[0]

}

resource "aws_s3_bucket" "root_bucket" {
  bucket = "${local.content_bucket_name}"
  acl    = "private"

  website {
    redirect_all_requests_to = "https://www.bayesai.io"
  }

  tags          = merge(var.tags, var.tags_s3_bucket_content, { Name = "${var.domain_name} Web Root Redirect" })
}

# DISABLE LOGGING
# resource "aws_s3_bucket" "logging" {
#   bucket        = "${var.domain_name}-logs"
#   force_destroy = var.force_destroy_buckets
#   grant {
#     id          = data.aws_canonical_user_id.current_user.id
#     type        = "CanonicalUser"
#     permissions = ["FULL_CONTROL"]
#   }
#   grant {
#     type        = "Group"
#     permissions = ["READ", "WRITE"]
#     uri         = "http://acs.amazonaws.com/groups/s3/LogDelivery"
#   }
#   grant {
#     # AWS Logs Delivery account: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/AccessLogs.html
#     id          = "c4c1ede66af53448b93c283ce9448c4ba468c9432aa01d700d3878632f77d2d0"
#     permissions = ["FULL_CONTROL"]
#     type        = "CanonicalUser"
#   }
#   tags = merge(var.tags, var.tags_s3_bucket_logging, { Name = "CloudFront logs for ${var.domain_name}" })
# }

resource "aws_s3_bucket" "content" {
  count         = var.create_content_bucket ? 1 : 0
  bucket        = "www.${local.content_bucket_name}"
  force_destroy = var.force_destroy_buckets
  tags          = merge(var.tags, var.tags_s3_bucket_content, { Name = "${var.domain_name} Static Content" })

  acl    = "private"
  
  policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
        "Sid": "AllowIPmix",
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": [
            "arn:aws:s3:::www.bayesai.io",
            "arn:aws:s3:::www.bayesai.io/*"
        ],
        "Condition": {
            "IpAddress": {
                "aws:SourceIp": [
                  "173.245.48.0/20",
                  "103.21.244.0/22",
                  "103.22.200.0/22",
                  "103.31.4.0/22",
                  "141.101.64.0/18",
                  "108.162.192.0/18",
                  "190.93.240.0/20",
                  "188.114.96.0/20",
                  "197.234.240.0/22",
                  "198.41.128.0/17",
                  "162.158.0.0/15",
                  "104.16.0.0/13",
                  "104.24.0.0/14",
                  "172.64.0.0/13",
                  "131.0.72.0/22",
                  "2400:cb00::/32",
                  "2606:4700::/32",
                  "2803:f800::/32",
                  "2405:b500::/32",
                  "2405:8100::/32",
                  "2a06:98c0::/29",
                  "2c0f:f248::/32"
              ]
            }
        }
    }
  ]
}
POLICY

  cors_rule {
    allowed_headers = ["Authorization", "Content-Length"]
    allowed_methods = ["GET", "POST"]
    allowed_origins = ["https://www.bayesai.io"]
    max_age_seconds = 3000
  }

  website {
      index_document = "index.html"
      error_document = "404.html"
    }
  }

data "aws_s3_bucket" "content" {
  count  = var.create_content_bucket ? 0 : 1
  bucket = "www.${local.content_bucket_name}"
}
