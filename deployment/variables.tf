variable "domain" {
  default = "policymanifolds.com"
}

variable "region" {
  default = "us-west-2"
}

variable "instance_type" {
  default = "t3.micro" # 2 vCPU, 1 GB — free tier eligible
}

variable "app_port" {
  default = 7860
}

variable "ssh_key_name" {
  description = "Name of the EC2 key pair for SSH access"
  default     = "policy-manifolds-key"
}

variable "ssh_public_key_path" {
  default = "~/.ssh/id_ed25519.pub"
}
