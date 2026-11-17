FROM node:20-alpine

WORKDIR /app

# Install npm dependencies (lockfile included for reproducible builds)
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# Copy application source
COPY . .

# Offline build
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
RUN npm run build

# Server-side proxy target used by next.config.js rewrites
ENV BACKEND_URL=http://backend:8000

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["npm", "start"]
