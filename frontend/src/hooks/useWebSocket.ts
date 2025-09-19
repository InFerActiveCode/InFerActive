// useWebSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketState, InferenceRequest, WebSocketResponse } from '../types/types';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8008/ws'

export const useWebSocket = () => {
  const [state, setState] = useState<WebSocketState>({
    connected: false,
    error: null,
    messages: [],
  });
  
  const wsRef = useRef<WebSocket | null>(null);
  
  const currentRequestIdRef = useRef<string | null>(null);

  const connect = useCallback(() => {
    try {
      wsRef.current = new WebSocket(WS_URL);

      wsRef.current.onopen = () => {
        console.log('WebSocket Connected');
        setState(prev => ({ ...prev, connected: true, error: null }));
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket Disconnected');
        setState(prev => ({ ...prev, connected: false }));
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
        setState(prev => ({ ...prev, error: 'WebSocket error occurred' }));
      };

      wsRef.current.onmessage = (event) => {
        const response = JSON.parse(event.data) as WebSocketResponse;
        console.log('Received:', response);
        
        if (response.request_id && !currentRequestIdRef.current) {
          console.log('Setting session request_id:', response.request_id);
          currentRequestIdRef.current = response.request_id;
        }
        
        setState(prev => ({
          ...prev,
          messages: [...prev.messages, {
            direction: 'received',
            timestamp: Date.now(),
            data: response
          }]
        }));
      };
    } catch (error) {
      console.error('Connection error:', error);
      setState(prev => ({ ...prev, error: 'Failed to connect' }));
    }
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
  }, []);

  const sendMessage = useCallback((message: InferenceRequest) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      if (currentRequestIdRef.current && !message.request_id) {
        message = {
          ...message,
          request_id: currentRequestIdRef.current
        };
      }
      
      console.log('Sending:', message);
      wsRef.current.send(JSON.stringify(message));
      setState(prev => ({
        ...prev,
        messages: [...prev.messages, {
          direction: 'sent',
          timestamp: Date.now(),
          data: message
        }]
      }));
    } else {
      setState(prev => ({ ...prev, error: 'WebSocket is not connected' }));
    }
  }, []);

  const clearMessages = useCallback(() => {
    currentRequestIdRef.current = null;
    setState(prev => ({ ...prev, messages: [] }));
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    state,
    sendMessage,
    connect,
    disconnect,
    clearMessages,
    getCurrentRequestId: () => currentRequestIdRef.current
  };
};