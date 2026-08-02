import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Settings, MessageSquare, Plus, Check, RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

const API_BASE = `${import.meta.env.BASE_URL}api/sessions`;

function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSessId, setActiveSessId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [logs, setLogs] = useState([]);
  
  const [inputText, setInputText] = useState('');
  const [activeTab, setActiveTab] = useState('chat'); // chat, prompt, global-prompt, logs
  const [globalPrompt, setGlobalPrompt] = useState('');
  const [isSavingGlobal, setIsSavingGlobal] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    fetchSessions();
    fetchGlobalPrompt();
    const interval = setInterval(fetchSessions, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (activeSessId) {
      fetchMessages();
      fetchPrompt();
      fetchLogs();
      const interval = setInterval(() => {
        fetchMessages();
        fetchLogs();
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [activeSessId]);

  const combinedLength = messages.length + logs.length;
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [combinedLength]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(API_BASE);
      const data = await res.json();
      if (Array.isArray(data)) {
        setSessions(data);
        setActiveSessId(current => {
          if (data.length > 0 && !current) {
            return data[0].id;
          }
          return current;
        });
      }
    } catch (e) {
      console.error(e);
    }
  };

  const createSession = async () => {
    try {
      const res = await fetch(API_BASE, { method: 'POST' });
      const data = await res.json();
      setActiveSessId(data.id);
      fetchSessions();
    } catch (e) {
      console.error(e);
    }
  };

  const fetchMessages = async () => {
    if (!activeSessId) return;
    try {
      const res = await fetch(`${API_BASE}/${activeSessId}/messages`);
      const data = await res.json();
      if (!data.error && Array.isArray(data)) setMessages(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchPrompt = async () => {
    if (!activeSessId) return;
    try {
      const res = await fetch(`${API_BASE}/${activeSessId}/prompt`);
      const data = await res.json();
      if (data.prompt) setPrompt(data.prompt);
    } catch (e) {
      console.error(e);
    }
  };

  const savePrompt = async () => {
    if (!activeSessId) return;
    setIsSaving(true);
    try {
      await fetch(`${API_BASE}/${activeSessId}/prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      setTimeout(() => setIsSaving(false), 1000);
    } catch (e) {
      console.error(e);
      setIsSaving(false);
    }
  };

  const fetchGlobalPrompt = async () => {
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}api/demon-prompt`);
      const data = await res.json();
      if (data.prompt) setGlobalPrompt(data.prompt);
    } catch (e) {
      console.error(e);
    }
  };

  const saveGlobalPrompt = async () => {
    setIsSavingGlobal(true);
    try {
      await fetch(`${import.meta.env.BASE_URL}api/demon-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: globalPrompt })
      });
      setTimeout(() => setIsSavingGlobal(false), 1000);
    } catch (e) {
      console.error(e);
      setIsSavingGlobal(false);
    }
  };

  const fetchLogs = async () => {
    if (!activeSessId) return;
    try {
      const res = await fetch(`${API_BASE}/${activeSessId}/logs`);
      const data = await res.json();
      if (Array.isArray(data)) setLogs(data);
    } catch (e) {
      console.error(e);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || !activeSessId) return;
    const msg = inputText;
    setInputText('');
    setMessages(prev => [...prev, { direction: 'in', content: { text: msg }, id: 'temp', timestamp: new Date().toISOString() }]);
    
    try {
      await fetch(`${API_BASE}/${activeSessId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: msg })
      });
    } catch (e) {
      console.error(e);
    }
  };

  const combinedTimeline = [...messages, ...logs].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-72 bg-slate-800 border-r border-slate-700 flex flex-col">
        <div className="p-6 border-b border-slate-700 flex justify-between items-center bg-slate-800">
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">Cartesian</h1>
          <button onClick={createSession} className="p-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg transition-colors">
            <Plus size={20} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-hide">
          {sessions.map(s => (
            <button
              key={s.id}
              onClick={() => setActiveSessId(s.id)}
              className={`w-full text-left p-4 rounded-xl transition-all duration-200 ${activeSessId === s.id ? 'bg-indigo-500/20 border border-indigo-500/50 shadow-lg shadow-indigo-500/10' : 'bg-slate-800/50 hover:bg-slate-700 border border-transparent'}`}
            >
              <div className="font-medium truncate">{s.id.replace('sess-', 'Sess-')}</div>
              <div className="text-xs text-slate-500 mt-1">{new Date(s.createdAt).toLocaleTimeString()}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative">
        {/* Header */}
        <header className="h-16 border-b border-slate-700/50 flex items-center px-6 glass absolute top-0 w-full z-10">
          <div className="flex space-x-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700 overflow-x-auto whitespace-nowrap">
            <button onClick={() => setActiveTab('chat')} className={`flex items-center px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'chat' ? 'bg-indigo-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}>
              <MessageSquare size={16} className="mr-2" /> Agent A Chat
            </button>
            <button onClick={() => setActiveTab('global-prompt')} className={`flex items-center px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'global-prompt' ? 'bg-indigo-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}>
              <Settings size={16} className="mr-2" /> Global Matrix Law
            </button>
            <button onClick={() => setActiveTab('prompt')} className={`flex items-center px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'prompt' ? 'bg-indigo-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}>
              <Settings size={16} className="mr-2" /> Session Override
            </button>
            <button onClick={() => setActiveTab('logs')} className={`flex items-center px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'logs' ? 'bg-indigo-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}>
              <Terminal size={16} className="mr-2" /> Demon Intercepts
            </button>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto mt-16 p-6 scrollbar-hide relative bg-gradient-to-br from-slate-900 to-slate-800">
          {activeTab === 'chat' && (
            <div className="max-w-4xl mx-auto flex flex-col h-full">
              <div className="flex-1 overflow-y-auto pb-32 space-y-6 scrollbar-hide">
                {combinedTimeline.map((item, i) => {
                  if (item.tool) {
                    // This is a log from Agent B
                    return (
                      <div key={i} className="flex justify-start">
                        <div className="max-w-[90%] rounded-2xl p-4 shadow-sm bg-slate-800/80 border border-amber-500/30 rounded-tl-sm relative overflow-hidden">
                          <div className="absolute top-0 left-0 w-1 h-full bg-amber-500"></div>
                          <div className="flex items-center space-x-2 text-amber-400 mb-2 font-mono text-xs font-bold uppercase tracking-wide">
                            <Terminal size={14} />
                            <span>Demon executed: {item.tool}</span>
                          </div>
                          {item.input && (
                            <div className="mb-2">
                              <span className="text-slate-400 text-xs">Input: </span>
                              <pre className="inline-block bg-slate-900 px-2 py-1 rounded text-emerald-400 font-mono text-xs border border-slate-700/50 break-all whitespace-pre-wrap">{JSON.stringify(item.input)}</pre>
                            </div>
                          )}
                          {item.reasoning && (
                            <div className="mb-2">
                              <span className="text-pink-400/80 text-xs mb-1 block">Demon Reasoning (Brain):</span>
                              <pre className="bg-slate-900/50 p-2 rounded text-pink-300/80 font-mono text-xs border border-pink-500/20 max-h-40 overflow-y-auto scrollbar-hide whitespace-pre-wrap italic">{item.reasoning}</pre>
                            </div>
                          )}
                          {item.output && (
                            <div>
                              <span className="text-slate-400 text-xs mb-1 block">Simulated Output:</span>
                              <pre className="bg-slate-900 p-2 rounded text-indigo-300 font-mono text-xs border border-slate-700/50 max-h-40 overflow-y-auto scrollbar-hide whitespace-pre-wrap">{typeof item.output === 'string' ? item.output : JSON.stringify(item.output, null, 2)}</pre>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  } else {
                    // This is a chat message (User or Agent A)
                    return (
                      <div key={i} className={`flex ${item.direction === 'in' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] rounded-2xl p-5 shadow-sm ${item.direction === 'in' ? 'bg-indigo-500/20 border border-indigo-500/30 text-indigo-100 rounded-tr-sm' : 'glass rounded-tl-sm'}`}>
                        <div className="prose prose-invert prose-indigo max-w-none text-[15px] leading-relaxed">
                          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                            {item.content.text}
                          </ReactMarkdown>
                        </div>
                      </div>
                      </div>
                    );
                  }
                })}
                <div ref={messagesEndRef} />
              </div>
              <div className="absolute bottom-6 left-6 right-6 flex justify-center">
                <form onSubmit={sendMessage} className="w-full max-w-4xl relative group">
                  <input
                    type="text"
                    value={inputText}
                    onChange={e => setInputText(e.target.value)}
                    placeholder="Ask Agent A..."
                    className="w-full bg-slate-800/80 border border-slate-600 rounded-full pl-6 pr-14 py-4 text-slate-200 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/20 shadow-lg backdrop-blur-md transition-all placeholder:text-slate-500"
                  />
                  <button type="submit" className="absolute right-2 top-2 p-2 bg-indigo-500 hover:bg-indigo-400 text-white rounded-full transition-transform active:scale-95 shadow-md">
                    <MessageSquare size={20} className="transform rotate-[-90deg] translate-x-[-1px] translate-y-[1px]" />
                  </button>
                </form>
              </div>
            </div>
          )}

          {activeTab === 'global-prompt' && (
            <div className="max-w-4xl mx-auto h-full flex flex-col pt-4">
              <div className="flex justify-between items-end mb-4">
                <div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">Global Matrix Law (Default Prompt)</h2>
                  <p className="text-slate-400 text-sm mt-1">This defines the core reality of the Cartesian Demon for ALL new sessions.</p>
                </div>
                <button onClick={saveGlobalPrompt} className={`flex items-center px-6 py-2.5 rounded-lg font-medium transition-all shadow-lg ${isSavingGlobal ? 'bg-pink-500 text-white' : 'bg-purple-600 hover:bg-purple-500 text-white'}`}>
                  {isSavingGlobal ? <Check size={18} className="mr-2" /> : <RefreshCw size={18} className="mr-2" />}
                  {isSavingGlobal ? 'Saved!' : 'Update Global Reality'}
                </button>
              </div>
              <textarea
                value={globalPrompt}
                onChange={e => setGlobalPrompt(e.target.value)}
                className="flex-1 w-full bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 text-slate-300 font-mono text-sm leading-relaxed focus:outline-none focus:border-pink-400 focus:ring-1 focus:ring-pink-400/30 shadow-inner resize-none scrollbar-hide"
              />
            </div>
          )}

          {activeTab === 'prompt' && (
            <div className="max-w-4xl mx-auto h-full flex flex-col pt-4">
              <div className="flex justify-between items-end mb-4">
                <div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">Session Override Prompt</h2>
                  <p className="text-slate-400 text-sm mt-1">Override the reality for THIS specific session only. Leave blank to use Global Law.</p>
                </div>
                <button onClick={savePrompt} className={`flex items-center px-6 py-2.5 rounded-lg font-medium transition-all shadow-lg ${isSaving ? 'bg-emerald-500 text-white' : 'bg-blue-600 hover:bg-blue-500 text-white'}`}>
                  {isSaving ? <Check size={18} className="mr-2" /> : <RefreshCw size={18} className="mr-2" />}
                  {isSaving ? 'Saved!' : 'Override Session'}
                </button>
              </div>
              <textarea
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="Leave blank to use the Global Matrix Law..."
                className="flex-1 w-full bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 text-slate-300 font-mono text-sm leading-relaxed focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-400/30 shadow-inner resize-none scrollbar-hide"
              />
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="max-w-5xl mx-auto">
              <div className="mb-6">
                <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-orange-400">Tool Intercepts</h2>
                <p className="text-slate-400 text-sm mt-1">Live view of A calling tools and B's simulated responses.</p>
              </div>
              <div className="space-y-4 pb-12">
                {logs.map((log, i) => (
                  <div key={i} className="glass rounded-xl overflow-hidden shadow-lg border-l-4 border-l-amber-500">
                    <div className="bg-slate-800/80 px-4 py-2 border-b border-slate-700/50 flex justify-between items-center">
                      <span className="font-mono font-bold text-amber-400 text-sm">{log.tool} Call</span>
                      <span className="text-xs text-slate-500 font-mono">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="p-4 grid grid-cols-1 gap-4">
                      {log.input && (
                        <div>
                          <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">Input from A</div>
                          <pre className="bg-slate-900 p-3 rounded-lg font-mono text-xs text-emerald-400 overflow-x-auto border border-slate-800">{JSON.stringify(log.input, null, 2)}</pre>
                        </div>
                      )}
                      {log.output && (
                        <div>
                          <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">Simulated Output from B</div>
                          <pre className="bg-slate-900 p-3 rounded-lg font-mono text-xs text-indigo-400 overflow-x-auto border border-slate-800">{JSON.stringify(log.output, null, 2)}</pre>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {logs.length === 0 && (
                  <div className="text-center py-20 text-slate-500 flex flex-col items-center">
                    <Terminal size={48} className="opacity-20 mb-4" />
                    <p>No tool intercepts yet for this session.</p>
                  </div>
                )}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
