provider "aws" {
  region = var.region
}

# --- SSH Key ---
resource "aws_key_pair" "app" {
  key_name   = var.ssh_key_name
  public_key = file(var.ssh_public_key_path)
}

# --- Security Group ---
resource "aws_security_group" "app" {
  name        = "policy-manifolds-sg"
  description = "Allow HTTP, HTTPS, and SSH"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "policy-manifolds" }
}

# --- Latest Ubuntu 22.04 AMI ---
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- EC2 Instance ---
resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.app.key_name
  vpc_security_group_ids = [aws_security_group.app.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    app_port = var.app_port
    domain   = var.domain
  })

  tags = { Name = "policy-manifolds" }
}

# --- Elastic IP ---
resource "aws_eip" "app" {
  instance = aws_instance.app.id
  domain   = "vpc"
  tags     = { Name = "policy-manifolds" }
}

# --- Route 53 DNS (enable after domain registration completes) ---
# Uncomment once `aws route53 list-hosted-zones` shows policymanifolds.com
#
# data "aws_route53_zone" "main" {
#   name         = "${var.domain}."
#   private_zone = false
# }
#
# resource "aws_route53_record" "root" {
#   zone_id = data.aws_route53_zone.main.zone_id
#   name    = var.domain
#   type    = "A"
#   ttl     = 300
#   records = [aws_eip.app.public_ip]
# }
#
# resource "aws_route53_record" "www" {
#   zone_id = data.aws_route53_zone.main.zone_id
#   name    = "www.${var.domain}"
#   type    = "CNAME"
#   ttl     = 300
#   records = [var.domain]
# }
