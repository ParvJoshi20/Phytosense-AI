'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, Send, X, Sparkles, User, AlertCircle, RefreshCw, Dna } from 'lucide-react';
import { DiagnosticResult } from '@/types/diagnosis';

interface AgronomistModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentResult: DiagnosticResult | null;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  time: string;
}

export const AgronomistModal: React.FC<AgronomistModalProps> = ({
  isOpen,
  onClose,
  currentResult,
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: `Hello! I am your Phytosense AI Virtual Agronomist & Plant Pathology Specialist. I can help with fungicide dosages, organic biological controls, FRAC rotation strategies, and weather adaptation for ${
        currentResult?.layer1.primaryDisease || 'your tomato crops'
      }. What questions do you have?`,
      time: 'Just now',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (textToSend?: string) => {
    const text = textToSend || input;
    if (!text.trim() || isLoading) return;

    const userMsg: Message = {
      role: 'user',
      content: text,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMsg],
          diagnosisContext: currentResult
            ? {
                primaryDisease: currentResult.layer1.primaryDisease,
                scientificName: currentResult.layer1.scientificName,
                severity: currentResult.layer1.severity,
                confidenceScore: currentResult.layer1.confidenceScore,
                pathogen: currentResult.layer1.pathogen,
                affectedCanopyEstimate: currentResult.layer1.affectedCanopyEstimate,
              }
            : null,
        }),
      });

      const data = await res.json();
      const botMsg: Message = {
        role: 'assistant',
        content: data.reply || 'Guidance received.',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            'I encountered an issue connecting to the pathology server. For Early Blight, recommended protocol is Copper Octanoate foliar spray (1 fl oz/gal) rotated with Chlorothalonil every 7 days.',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const quickPrompts = [
    'What is the exact dosage per gallon?',
    'What organic fungicides are OMRI listed?',
    'How do I prevent FRAC fungicide resistance?',
  ];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        className="w-full max-w-2xl h-[560px] rounded-3xl bg-[#121215] border border-[#5B6987]/30 shadow-2xl flex flex-col overflow-hidden"
      >
        {/* Header */}
        <div className="p-4 border-b border-[#5B6987]/20 flex items-center justify-between bg-[#121215]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-[#2A7FFF]/20 border border-[#2A7FFF]/40 flex items-center justify-center text-[#2A7FFF]">
              <Bot className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-[#E2E8F0] flex items-center gap-2">
                Phytosense AI Agronomist Specialist
              </h3>
              <p className="text-[10px] font-mono text-[#828C9E]">
                {currentResult
                  ? `Context: ${currentResult.layer1.primaryDisease} (${currentResult.layer1.severity})`
                  : 'General Pathology Consulting'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[#828C9E] hover:text-[#E2E8F0] hover:bg-white/5 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Message Feed */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3.5 bg-[#08080A]/60">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex gap-2.5 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {m.role === 'assistant' && (
                <div className="w-7 h-7 rounded-lg bg-[#2A7FFF]/20 border border-[#2A7FFF]/40 flex items-center justify-center text-[#2A7FFF] shrink-0 mt-0.5">
                  <Bot className="w-3.5 h-3.5" />
                </div>
              )}
              <div
                className={`max-w-[82%] p-3.5 rounded-2xl text-xs leading-relaxed ${
                  m.role === 'user'
                    ? 'bg-[#2A7FFF] text-[#08080A] font-medium'
                    : 'bg-[#121215] border border-[#5B6987]/30 text-[#E2E8F0]'
                }`}
              >
                <div className="whitespace-pre-wrap">{m.content}</div>
                <div
                  className={`text-[9px] font-mono mt-1 ${
                    m.role === 'user' ? 'text-[#08080A]/70 text-right' : 'text-[#828C9E]'
                  }`}
                >
                  {m.time}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-2.5 items-center text-xs text-[#828C9E] font-mono p-2">
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#2A7FFF]" />
              <span>Agronomist is formulating pathological prescription...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick prompt chips */}
        <div className="p-2.5 bg-[#121215] border-t border-[#5B6987]/15 flex items-center gap-1.5 overflow-x-auto">
          {quickPrompts.map((qp, i) => (
            <button
              key={i}
              onClick={() => handleSend(qp)}
              className="px-2.5 py-1 rounded-lg text-[10px] font-mono text-[#828C9E] hover:text-[#2A7FFF] bg-[#08080A] hover:bg-[#2A7FFF]/10 border border-[#5B6987]/30 hover:border-[#2A7FFF]/40 shrink-0 transition-all"
            >
              {qp}
            </button>
          ))}
        </div>

        {/* Input box */}
        <div className="p-3 bg-[#121215] border-t border-[#5B6987]/20 flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about treatment dosages, FRAC rotations, or spray timing..."
            className="flex-1 px-4 py-2.5 rounded-xl bg-[#08080A] border border-[#5B6987]/30 text-xs text-[#E2E8F0] placeholder-[#828C9E] focus:outline-none focus:border-[#2A7FFF]"
          />
          <motion.button
            whileTap={{ scale: 0.94 }}
            onClick={() => handleSend()}
            disabled={!input.trim() || isLoading}
            className="p-2.5 rounded-xl bg-[#2A7FFF] disabled:opacity-50 text-[#08080A] hover:bg-[#2A7FFF]/90 transition-all"
          >
            <Send className="w-4 h-4" />
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
};
