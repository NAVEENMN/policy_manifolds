#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

export DEBIAN_FRONTEND=noninteractive

# --- System packages ---
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3.11-dev nginx certbot python3-certbot-nginx git

# --- App user ---
useradd -m -s /bin/bash appuser || true

# --- Clone / deploy app ---
APP_DIR=/opt/policy-manifolds
mkdir -p $APP_DIR
chown appuser:appuser $APP_DIR

# Placeholder: app code will be deployed via rsync/scp after instance is up
cat > $APP_DIR/setup.sh << 'SETUP'
#!/bin/bash
set -euo pipefail
cd /opt/policy-manifolds
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
SETUP
chmod +x $APP_DIR/setup.sh

# --- Nginx reverse proxy ---
cat > /etc/nginx/sites-available/policy-manifolds << 'NGINX'
server {
    listen 80;
    server_name ${domain} www.${domain};

    location / {
        proxy_pass http://127.0.0.1:${app_port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/policy-manifolds /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# --- Systemd service for app ---
cat > /etc/systemd/system/policy-manifolds.service << 'SERVICE'
[Unit]
Description=Policy Manifolds App
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/policy-manifolds
ExecStart=/opt/policy-manifolds/venv/bin/python app.py --port ${app_port}
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
# Service will be enabled after app code is deployed

echo "User data complete. Deploy app code to /opt/policy-manifolds then run:"
echo "  sudo -u appuser bash /opt/policy-manifolds/setup.sh"
echo "  sudo systemctl enable --now policy-manifolds"
echo "  sudo certbot --nginx -d ${domain} -d www.${domain} --non-interactive --agree-tos -m YOUR_EMAIL@example.com"
