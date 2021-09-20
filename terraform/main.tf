resource "aws_s3_bucket" "content" {
  bucket = "bayesai.io-prod-content-bucket"
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

  ## Other bucket policies... because otherwise you could just let the module create this policy
  #statement {
  #  actions   = ["s3:GetObject"]
  #}
}
resource "aws_s3_bucket_policy" "content" {
  bucket = aws_s3_bucket.content.id
  policy = data.aws_iam_policy_document.content.json
}

module "external_bucket_static_site" {
  source = "modules/static-site"
  domain_name = "bayesai.io"

  acm_certificate_arn = "arn:aws:acm:us-east-1:123412341234:certificate/1234abcd-1234-abcd-1234-abcd1234abcd"

  # Optional
  hosted_zone_id               = "Z0123456789ABCDEFGHIJ"
  default_subdirectory_object  = "index.html"
  create_content_bucket        = false
  manage_content_bucket_policy = false
  content_bucket_name          = aws_s3_bucket.content.id
}

