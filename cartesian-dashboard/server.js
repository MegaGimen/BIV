import express from 'express';
import { Database } from 'bun:sqlite';
import fs from 'fs';
import path from 'path';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

const SESSIONS_BASE = '/home/BIV/data/v2-sessions';
const GROUP_ID = 'ag-1785342819022-telo5r'; 
const GLOBAL_PROMPT_PATH = '/home/BIV/data/global_demon_prompt.txt';

app.get('/api/demon-prompt', (req, res) => {
  try {
    if (fs.existsSync(GLOBAL_PROMPT_PATH)) {
      res.json({ prompt: fs.readFileSync(GLOBAL_PROMPT_PATH, 'utf-8') });
    } else {
      res.json({ prompt: '' });
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/demon-prompt', (req, res) => {
  try {
    fs.writeFileSync(GLOBAL_PROMPT_PATH, req.body.prompt || '');
    res.json({ success: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/sessions', (req, res) => {
  const groupPath = path.join(SESSIONS_BASE, GROUP_ID);
  if (!fs.existsSync(groupPath)) return res.json([]);
  
  const sessions = fs.readdirSync(groupPath)
    .filter(dir => dir.startsWith('sess-'))
    .map(dir => {
      const p = path.join(groupPath, dir);
      const stat = fs.statSync(p);
      return { id: dir, createdAt: stat.birthtime };
    })
    .sort((a, b) => b.createdAt - a.createdAt);
    
  res.json(sessions);
});

app.post('/api/sessions', async (req, res) => {
  const newSessId = `sess-${Date.now()}-ui`;
  const sessPath = path.join(SESSIONS_BASE, GROUP_ID, newSessId);
  fs.mkdirSync(sessPath, { recursive: true });

  const inboundDbPath = path.join(sessPath, 'inbound.db');
  const outboundDbPath = path.join(sessPath, 'outbound.db');
  
  try {
    const dbIn = new Database(inboundDbPath);
    dbIn.run(`CREATE TABLE messages_in (id TEXT PRIMARY KEY, seq INTEGER UNIQUE, kind TEXT NOT NULL, timestamp TEXT NOT NULL, status TEXT DEFAULT 'pending', process_after TEXT, recurrence TEXT, series_id TEXT, tries INTEGER DEFAULT 0, trigger INTEGER NOT NULL DEFAULT 1, platform_id TEXT, channel_type TEXT, thread_id TEXT, content TEXT NOT NULL, source_session_id TEXT, on_wake INTEGER NOT NULL DEFAULT 0)`);
    dbIn.run(`CREATE TABLE delivered (message_out_id TEXT PRIMARY KEY, platform_message_id TEXT, status TEXT NOT NULL DEFAULT 'delivered', delivered_at TEXT NOT NULL)`);
    dbIn.run(`CREATE TABLE destinations (name TEXT PRIMARY KEY, display_name TEXT, type TEXT NOT NULL, channel_type TEXT, platform_id TEXT, agent_group_id TEXT)`);
    dbIn.run(`CREATE TABLE session_routing (id INTEGER PRIMARY KEY CHECK (id = 1), channel_type TEXT, platform_id TEXT, thread_id TEXT)`);

    const dbOut = new Database(outboundDbPath);
    dbOut.run(`CREATE TABLE messages_out (id TEXT PRIMARY KEY, seq INTEGER UNIQUE, in_reply_to TEXT, timestamp TEXT NOT NULL, deliver_after TEXT, recurrence TEXT, kind TEXT NOT NULL, platform_id TEXT, channel_type TEXT, thread_id TEXT, content TEXT NOT NULL)`);
    dbOut.run(`CREATE TABLE processing_ack (message_id TEXT PRIMARY KEY, status TEXT NOT NULL, status_changed TEXT NOT NULL)`);
    dbOut.run(`CREATE TABLE session_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)`);
    dbOut.run(`CREATE TABLE container_state (id INTEGER PRIMARY KEY CHECK (id = 1), current_tool TEXT, tool_declared_timeout_ms INTEGER, tool_started_at TEXT, updated_at TEXT NOT NULL)`);
    
    dbIn.run(`INSERT INTO session_routing (id, channel_type, platform_id, thread_id) VALUES (1, 'cli', 'local', ?)`, [newSessId]);
    dbIn.close();
    dbOut.close();

    const v2DbPath = '/home/BIV/data/v2.db';
    const v2Db = new Database(v2DbPath);
    v2Db.run(`INSERT INTO sessions (id, agent_group_id, status, created_at) VALUES (?, ?, 'active', ?)`, [newSessId, GROUP_ID, new Date().toISOString()]);
    v2Db.close();

    res.json({ id: newSessId });
  } catch(e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/sessions/:id/messages', (req, res) => {
  const sessPath = path.join(SESSIONS_BASE, GROUP_ID, req.params.id);
  if (!fs.existsSync(sessPath)) return res.status(404).json({ error: 'Not found' });

  let messages = [];
  try {
    const inboundDb = new Database(path.join(sessPath, 'inbound.db'), { readonly: true });
    const inRows = inboundDb.query('SELECT id, timestamp, status, content, kind FROM messages_in ORDER BY timestamp ASC').all();
    if (inRows) messages.push(...inRows.map(r => ({ ...r, direction: 'in', content: JSON.parse(r.content) })));
    inboundDb.close();

    const outboundDb = new Database(path.join(sessPath, 'outbound.db'), { readonly: true });
    const outRows = outboundDb.query('SELECT id, timestamp, content, kind FROM messages_out ORDER BY timestamp ASC').all();
    if (outRows) messages.push(...outRows.map(r => ({ ...r, direction: 'out', content: JSON.parse(r.content) })));
    outboundDb.close();
  } catch (e) {
    console.error(e);
  }

  messages.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  res.json(messages);
});

app.post('/api/sessions/:id/messages', (req, res) => {
  const sessPath = path.join(SESSIONS_BASE, GROUP_ID, req.params.id);
  try {
    const inboundDb = new Database(path.join(sessPath, 'inbound.db'));
    const msgId = `send-${Date.now()}-ui`;
    const content = JSON.stringify({ text: req.body.text });
    
    inboundDb.run(
      `INSERT INTO messages_in (id, kind, timestamp, status, trigger, channel_type, platform_id, content) VALUES (?, 'chat', ?, 'pending', 1, 'cli', 'local', ?)`,
      [msgId, new Date().toISOString(), content]
    );
    inboundDb.close();
    res.json({ id: msgId });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/sessions/:id/prompt', (req, res) => {
  const sessPath = path.join(SESSIONS_BASE, GROUP_ID, req.params.id);
  const p = path.join(sessPath, 'demon_prompt.txt');
  if (fs.existsSync(p)) {
    res.json({ prompt: fs.readFileSync(p, 'utf-8') });
  } else {
    res.json({ prompt: '' });
  }
});

app.post('/api/sessions/:id/prompt', (req, res) => {
  const sessPath = path.join(SESSIONS_BASE, GROUP_ID, req.params.id);
  const p = path.join(sessPath, 'demon_prompt.txt');
  if (!req.body.prompt || req.body.prompt.trim() === '') {
    if (fs.existsSync(p)) fs.unlinkSync(p);
  } else {
    fs.writeFileSync(p, req.body.prompt);
  }
  res.json({ success: true });
});

app.get('/api/sessions/:id/logs', (req, res) => {
  const sessPath = path.join(SESSIONS_BASE, GROUP_ID, req.params.id);
  const p = path.join(sessPath, 'demon_logs.jsonl');
  if (fs.existsSync(p)) {
    const lines = fs.readFileSync(p, 'utf-8').split('\n').filter(l => l.trim());
    res.json(lines.map(l => JSON.parse(l)));
  } else {
    res.json([]);
  }
});

app.listen(3033, '0.0.0.0', () => {
  console.log('Backend listening on 0.0.0.0:3033');
});
