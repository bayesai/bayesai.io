output "content_bucket_id" {
  value       = local.content_bucket.id
  description = "The id of the S3 Bucket where you will put your static html content.  CloudFront will handle requests and fetch data from this bucket.  Depending on your settings, this could be made by the module, or it could be the bucket ID passed in by variable"
}
output "content_bucket_arn" {
  value       = local.content_bucket.arn
  description = "The ARN of the S3 Bucket where you will put your static html content.  CloudFront will handle requests and fetch data from this bucket.  Depending on your settings, this could be made by the module, or it could be the bucket ID passed in by variable"
}

