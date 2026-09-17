FROM node:20-alpine
WORKDIR /app

COPY --chown=node:node package.json ./
RUN npm install --omit=dev

COPY --chown=node:node src ./src
COPY --chown=node:node public ./public

USER node

EXPOSE 3000
CMD ["node", "src/server.js"]
