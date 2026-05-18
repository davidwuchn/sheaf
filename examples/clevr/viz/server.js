#!/usr/bin/env node
// CLEVR visualizer: tiny Node HTTP server, no deps.
// Serves index.html, streams `sheaf train.shf` stdout via SSE,
// runs `sheaf run.shf` on demand.

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 8080;
const CLEVR_DIR = path.resolve(__dirname, '..');
const INDEX_PATH = path.join(__dirname, 'index.html');

let trainingProc = null;

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(fs.readFileSync(INDEX_PATH, 'utf8'));
    return;
  }

  if (req.method === 'POST' && req.url === '/train') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });
    if (trainingProc) trainingProc.kill();
    trainingProc = spawn('sheaf', ['train.shf'], { cwd: CLEVR_DIR });
    const sendLine = (line) => res.write(`data: ${line}\n\n`);
    let buf = '';
    trainingProc.stdout.on('data', (chunk) => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) sendLine(line);
    });
    trainingProc.on('close', () => {
      if (buf) sendLine(buf);
      res.write('event: done\ndata: \n\n');
      res.end();
      trainingProc = null;
    });
    req.on('close', () => {
      if (trainingProc) { trainingProc.kill(); trainingProc = null; }
    });
    return;
  }

  if (req.method === 'POST' && req.url === '/train/stop') {
    if (trainingProc) { trainingProc.kill(); trainingProc = null; }
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('stopped');
    return;
  }

  if (req.method === 'POST' && req.url === '/run') {
    const proc = spawn('sheaf', ['run.shf'], { cwd: CLEVR_DIR });
    let out = '';
    proc.stdout.on('data', (c) => out += c.toString());
    proc.stderr.on('data', (c) => out += c.toString());
    proc.on('close', () => {
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end(out);
    });
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

server.listen(PORT, () => {
  console.log(`Dashboard at http://localhost:${PORT}`);
});
