variable "region" {
  description = "The region from which this module will be executed"
  type        = string
  default     = "ap-southeast-1"
}

variable "role_name" {
  description = "The name of the role. It will forces new resource on change."
  type        = string
}

variable "role_path" {
  description = "The path to the role. See https://docs.aws.amazon.com/IAM/latest/UserGuide/Using_Identifiers.html for more information."
  type        = string
  default     = "/"
}

variable "role_description" {
  description = "The description of the role."
  type        = string
}

variable "project" {
  description = "Abbreviation of the product domain the created resources belong to"
  type        = string
}

variable "owner" {
  description = "Name of the owner of the resource. Contact them if you have issues"
  type        = string
}

variable "environment" {
  type        = string
  description = "Will be used in Environment tag"
}

variable "role_tags" {
  description = "Additional tags to be put on iam role"
  type        = map(string)
  default     = {}
}

variable "role_assume_policy" {
  description = "IAM policy document that grants an entity permission to assume the role in JSON format."
  type        = string
}

variable "role_force_detach_policies" {
  description = "Specifies to force detaching any policies the role has before destroying it."
  default     = false
}

variable "role_max_session_duration" {
  description = "The maximum session duration (in seconds) that you want to set for the specified role. If you do not specify a value for this setting, the default maximum of one hour is applied. This setting can have a value from 1 hour to 12 hours."
  default     = 3600
}

variable "role_permission_boundary" {
  description = "IAM policy ARN limiting the maximum access this role can have"
  type        = string
  default     = ""
}

