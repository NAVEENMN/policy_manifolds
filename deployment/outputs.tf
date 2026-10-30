output "instance_ip" {
  value = aws_eip.app.public_ip
}

output "instance_id" {
  value = aws_instance.app.id
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/id_ed25519 ubuntu@${aws_eip.app.public_ip}"
}

output "deploy_command" {
  value = "rsync -avz --exclude venv --exclude __pycache__ --exclude .git -e 'ssh -i ~/.ssh/id_ed25519' ../ ubuntu@${aws_eip.app.public_ip}:/opt/policy-manifolds/"
}

output "url" {
  value = "https://${var.domain}"
}
