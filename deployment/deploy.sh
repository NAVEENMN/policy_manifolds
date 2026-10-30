#!/bin/bash
# Deploy app code to EC2 instance after terraform apply
set -euo pipefail
cd "$(dirname "$0")/.."

IP=$(cd deployment && terraform output -raw instance_ip)
SSH="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ubuntu@$IP"

echo "Deploying to $IP..."

# Sync app code
rsync -avz --delete \
  --exclude venv \
  --exclude __pycache__ \
  --exclude .git \
  --exclude 'remote_results' \
  --exclude 'surface_results' \
  --exclude '*.zip' \
  --include 'models/***' \
  --include 'policy-manifolds/***' \
  --include 'app.py' \
  --include 'requirements.txt' \
  --include 'export_policy.py' \
  --include 'export_pendulum.py' \
  --include 'train_kan_ppo.py' \
  --include 'pendulum_kan.py' \
  --filter '- *' \
  -e "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
  ./ ubuntu@$IP:/tmp/app-staging/

# Move to app dir and set up
$SSH "sudo rsync -a /tmp/app-staging/ /opt/policy-manifolds/ && sudo chown -R appuser:appuser /opt/policy-manifolds"
$SSH "sudo -u appuser bash /opt/policy-manifolds/setup.sh"
$SSH "sudo systemctl enable --now policy-manifolds"

echo ""
echo "App deployed! Setting up HTTPS..."
$SSH "sudo certbot --nginx -d policymanifolds.com -d www.policymanifolds.com --non-interactive --agree-tos -m \$CERTBOT_EMAIL" || echo "Certbot failed — DNS may not have propagated yet. Run manually later."

echo ""
echo "Done! Visit: https://policymanifolds.com"
