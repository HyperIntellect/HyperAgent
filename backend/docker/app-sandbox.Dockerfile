# Pre-built app sandbox image with cached Node.js templates.
#
# Scaffolds common templates at build time so that scaffold_project()
# can do a fast `cp -a` (~2-3s) instead of running npm from scratch
# (~60-300s on a cold cache).
#
# Build:
#   docker build -f backend/docker/app-sandbox.Dockerfile \
#       -t hyperagent/app-sandbox:latest backend/docker/
#
# Templates cached under /opt/templates/<name>/ with node_modules.
# A marker file at /opt/templates/.cached lists what was pre-scaffolded.

FROM node:20-slim

# git is required by create-next-app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV TEMPLATE_DIR=/opt/templates

RUN mkdir -p $TEMPLATE_DIR

# ---- React (Vite 5) ----
RUN cd /tmp \
    && npm create vite@5 app -- --template react \
    && cd app && npm install \
    && mv /tmp/app $TEMPLATE_DIR/react \
    && rm -rf /tmp/app

# ---- React TypeScript (Vite 5) ----
RUN cd /tmp \
    && npm create vite@5 app -- --template react-ts \
    && cd app && npm install \
    && mv /tmp/app $TEMPLATE_DIR/react-ts \
    && rm -rf /tmp/app

# ---- Vue 3 (Vite 5) ----
RUN cd /tmp \
    && npm create vite@5 app -- --template vue \
    && cd app && npm install \
    && mv /tmp/app $TEMPLATE_DIR/vue \
    && rm -rf /tmp/app

# ---- Next.js 14 ----
RUN cd /tmp \
    && npx create-next-app@14 app --typescript --tailwind --eslint --app --src-dir --use-npm --no-git \
    && mv /tmp/app $TEMPLATE_DIR/nextjs \
    && rm -rf /tmp/app

# ---- Express ----
RUN mkdir -p /tmp/app \
    && cd /tmp/app && npm init -y && npm install express \
    && mv /tmp/app $TEMPLATE_DIR/express \
    && rm -rf /tmp/app

# Clean npm cache to reduce image size
RUN npm cache clean --force

# Write marker file listing cached templates
RUN ls -1 $TEMPLATE_DIR > $TEMPLATE_DIR/.cached

WORKDIR /home/user
