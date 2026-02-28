# VPS Deployment Guide

Reusable pattern: FastAPI + Caddy + systemd + GitHub Actions on a VPS (Ubuntu 24.04).

---

## 1. Provision the VPS (DigitalOcean)

### Via DigitalOcean Cloud Console

1. Go to https://cloud.digitalocean.com → Create → Droplets
2. **Configure Droplet**:
   - Region: **New York (NYC1)** or San Francisco — whichever is closer
   - Image: **Ubuntu 24.04 LTS**
   - Size: Shared CPU → Basic → **$6/mo** (1 vCPU, 2 GB RAM, 50 GB disk)
   - Authentication: **SSH Key** (add yours under Settings → Security first)
   - Hostname: `transcribe-api`
3. Click **Create Droplet**

### Generate SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Paste this into DigitalOcean's SSH key field
```

### First connection

```bash
ssh root@YOUR_SERVER_IP
```

---

## 2. Server Hardening

Run these commands on the server as root:

### 2a. System updates

```bash
apt update && apt upgrade -y
apt install -y ufw fail2ban curl git python3-pip python3-venv
```

### 2b. Create a deploy user (don't run your app as root)

```bash
adduser --disabled-password --gecos "" deploy
mkdir -p /home/deploy/.ssh
cp ~/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys

# Give deploy user sudo access
usermod -aG sudo deploy
echo "deploy ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/deploy
```

### 2c. Disable root login and password auth

Edit `/etc/ssh/sshd_config`:

```
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
```

Then:

```bash
systemctl restart sshd
```

**Test in a new terminal before closing your root session:**

```bash
ssh deploy@YOUR_SERVER_IP
```

### 2d. Firewall (UFW)

```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP (Caddy redirect)
ufw allow 443/tcp   # HTTPS
ufw enable
```

### 2e. Fail2ban (brute-force protection)

```bash
systemctl enable fail2ban
systemctl start fail2ban
```

Default config protects SSH. Good enough for now.

---

## 3. Install Caddy

```bash
apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt update
apt install caddy
```

Copy the `Caddyfile` to `/etc/caddy/Caddyfile` (see file in this repo), then:

```bash
systemctl enable caddy
systemctl restart caddy
```

---

## 4. Deploy the Application

As the `deploy` user:

```bash
sudo -u deploy -i

# Clone your repo
git clone https://github.com/YOUR_USER/YOUR_REPO.git /home/deploy/app
cd /home/deploy/app

# Create virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn python-dotenv python-multipart

# Create .env from template
cp .env.example .env
nano .env  # Fill in your actual keys
```

### Install the systemd service

```bash
sudo cp deploy/transcribe-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable transcribe-api
sudo systemctl start transcribe-api

# Check it's running
sudo systemctl status transcribe-api
sudo journalctl -u transcribe-api -f  # Live logs
```

---

## 5. DNS Setup

Point your domain to the server IP:

```
A record:  api.yourdomain.com → YOUR_SERVER_IP
```

Caddy will auto-provision HTTPS via Let's Encrypt once DNS propagates.

---

## 6. Test It

```bash
# Health check
curl https://api.yourdomain.com/health

# Transcribe + translate (Italian → English, using OpenAI Whisper)
curl -X POST https://api.yourdomain.com/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "video=@video.mp4" \
  -F "source_language=it" \
  -F "target_language=en" \
  -F 'services=["openai"]' \
  -F "translate=true"

# Transcribe only (no translation)
curl -X POST https://api.yourdomain.com/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "video=@audio.mp3" \
  -F "source_language=ru" \
  -F 'services=["google"]' \
  -F "translate=false"

# Multi-service transcribe (both OpenAI and Google concurrently)
curl -X POST https://api.yourdomain.com/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "video=@video.mp4" \
  -F "source_language=ar" \
  -F "target_language=en" \
  -F 'services=["openai","google"]'

# Translate SRT text
curl -X POST https://api.yourdomain.com/translate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "text=1\n00:00:00,000 --> 00:00:02,000\nCiao mondo" \
  -F "source_language=it" \
  -F "target_language=en"
```

---

## 7. GitHub Actions Auto-Deploy

The workflow at `.github/workflows/deploy.yml` will SSH into your server and pull + restart on every push to `main`.

### Required GitHub Secrets

Set these in your repo → Settings → Secrets and variables → Actions:

| Secret | Value |
|--------|-------|
| `VPS_HOST` | Your server IP or domain |
| `VPS_USER` | `deploy` |
| `VPS_SSH_KEY` | Contents of `~/.ssh/id_ed25519` (private key) |

---

## 8. Maintenance Commands

```bash
# View logs
sudo journalctl -u transcribe-api -f

# Restart after config change
sudo systemctl restart transcribe-api

# Manual deploy
cd /home/deploy/app && git pull && source venv/bin/activate && pip install -r requirements.txt
sudo systemctl restart transcribe-api

# Check Caddy logs
sudo journalctl -u caddy -f
```

---

## Reusing This Pattern

For a new project, copy these files into your repo:
- `api_server.py` — adapt endpoints
- `deploy/Caddyfile` — change domain
- `deploy/transcribe-api.service` — change service name and paths
- `.github/workflows/deploy.yml` — change service name
- `.env.example` — change variables

The server setup (steps 1–3) only needs to happen once per VPS.
