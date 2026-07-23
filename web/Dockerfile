# Multi-stage build for TanStack Start web app
FROM node:20-alpine AS build
WORKDIR /app

# Copy dependency manifests and install dependencies
COPY package.json package-lock.json* ./
RUN npm ci || npm install

# Copy source and build
COPY . .
RUN npm run build

# Production stage — serve pre-built static output via Nginx
FROM nginx:1.27-alpine
COPY --from=build /app/.output/public /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
