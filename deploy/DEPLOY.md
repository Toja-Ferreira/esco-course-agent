# Deploying on the university VM

The assistant runs in Docker next to POF, listens only on `127.0.0.1:8090`, and
nginx serves it at `/esco/` on POF's address. POF's tab
(`/pof/esco-assistant`) shows it in a frame; logins are POF's.

Paths below assume POF at `<base-dir>/Pof` and this repo next to it.
Nothing here touches the other projects on the server.

## 0. POF first

1. Merge PR #8 on GitHub.
2. On the VM, update and redeploy POF **the way POF is normally deployed**
   (ask Claude Code in the POF repo if unsure: "how do I deploy main on the VM?"):
   ```bash
   cd <base-dir>/Pof && git pull
   ```
3. Apply the migration (adds the `esco_test` role; touches no data):
   ```bash
   docker exec pof-backend python manage.py migrate
   docker exec pof-backend python manage.py showmigrations api | tail -3   # 0018 has [X]
   ```

## 1. Get the code and the data

```bash
cd <base-dir>
git clone https://github.com/Toja-Ferreira/esco-course-agent.git
cd esco-course-agent
mkdir -p data results hf-cache
```

From **your laptop**, copy the ESCO data (5 parquet files + the `.npy` skill vectors):
```bash
scp ~/esco-course-agent/data/* <user>@<vm-host>:<base-dir>/esco-course-agent/data/
```

## 2. Settings

```bash
cp .env.example .env
nano .env
```
Fill in `DEEPSEEK_API_KEY` and `HF_TOKEN`, and add your user's ids:
```bash
echo "APP_UID=$(id -u)" >> .env; echo "APP_GID=$(id -g)" >> .env
chmod 600 .env
```
`HOST`, `PORT`, `ESCO_AUTH`, `POF_API_URL` and `ESCO_DATA_DIR` are set by
`docker-compose.yml` and need no change. Leave the `UA_OAUTH_*` lines empty.

## 3. Start

```bash
docker compose up -d --build      # first build: several minutes (torch)
docker compose logs -f            # Ctrl+C to stop following
```

Checks:
```bash
curl -s 127.0.0.1:8090/api/me; echo                    # {"mode": "pof", ...}
# POF's backend reachable from the container (expect "HTTP Error 401"):
docker exec esco-assistant python -c "import urllib.request as u; u.urlopen('http://host.docker.internal:5181/api/auth/user/')"
# download the embedding model now, not on the first teacher's message:
docker exec esco-assistant python -c "from esco_core.dense import MODEL_ID, MODEL_REVISION; from sentence_transformers import SentenceTransformer; SentenceTransformer(MODEL_ID, revision=MODEL_REVISION); print('model ok')"
```
If the second check times out instead of giving 401, the firewall blocks
Docker from reaching the host. The fix is to join POF's Docker network and use
`POF_API_URL: http://<pof-backend-host>:<port>` instead.

## 4. nginx

```bash
sudo cp /etc/nginx/sites-enabled/pof ~/pof.nginx.bak
sudo nano /etc/nginx/sites-enabled/pof
```
Inside the `server { ... }` block, next to the other `location` blocks, add:
```nginx
    location = /esco { return 301 /esco/; }
    location ^~ /esco/ {
        proxy_pass http://127.0.0.1:8090/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;          # replies are streamed
        proxy_read_timeout 300s;
    }
```
```bash
sudo nginx -t && sudo systemctl reload nginx
curl -sI http://<vm-host>/esco/ | grep -iE "^HTTP|x-frame"   # 200 or 302, SAMEORIGIN
```
(`nginx -t` failing changes nothing; fix the typo, or restore the backup.)

## 5. End-to-end test

1. Log in to POF with your UA account (administrator) → sidebar → Assistente ESCO.
   Do one short session to the closing questions.
2. Give the `esco_test` role to a second account (POF's user admin page) and
   log in with it: only the assistant tab, other POF pages bounce back.
3. If the UA login itself fails after the POF update: `git revert 455dfdb`
   in POF (the OAuth state check) and redeploy POF.
4. Sessions appear in `results/sessions/` with `"role"` set; delete your test
   sessions before the study starts.

## 6. Backups (sessions and the salt)

```bash
mkdir -p <base-dir>/esco-backups
crontab -e
```
Add:
```
15 3 * * * tar czf <base-dir>/esco-backups/sessions-$(date +\%F).tgz -C <base-dir>/esco-course-agent/results sessions .salt
```
Keep the salt with the sessions: without it, a teacher's sessions can no longer
be linked to each other.

## Later

- Update: `git pull && docker compose up -d --build`.
- Stop: `docker compose down` (sessions stay in `results/`).
- Remove completely: `docker compose down`, delete the nginx block, `sudo systemctl reload nginx`.
- The daily model-turn cap is `ESCO_DAILY_LIMIT` in `.env` (default 400).
