import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketState, WebSocketRequest, WebSocketResponse } from '../types/types';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8008/ws';

export const useWebSocket = (autoConnect: boolean = true) => {
  const [state, setState] = useState<WebSocketState>({
    connected: false,
    error: null,
    messages: [],
  });

  const wsRef = useRef<WebSocket | null>(null);

  const currentRequestIdRef = useRef<string | null>(null);

  const connect = useCallback(() => {
    if (!autoConnect) return;

    try {
      wsRef.current = new WebSocket(WS_URL);

      wsRef.current.onopen = () => {
        setState(prev => ({ ...prev, connected: true, error: null }));
      };

      wsRef.current.onclose = () => {
        setState(prev => ({ ...prev, connected: false }));
      };

      wsRef.current.onerror = () => {
        setState(prev => ({ ...prev, error: 'WebSocket error occurred' }));
      };

      wsRef.current.onmessage = (event) => {
        let response: WebSocketResponse;
        try {
          response = JSON.parse(event.data) as WebSocketResponse;
        } catch {
          setState(prev => ({ ...prev, error: 'Received invalid backend message' }));
          return;
        }

        if (response.request_id && !currentRequestIdRef.current) {
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
    } catch {
      setState(prev => ({ ...prev, error: 'Failed to connect' }));
    }
  }, [autoConnect]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
  }, []);

  const sendMessage = useCallback((message: WebSocketRequest) => {
    if (!autoConnect) {
      setState(prev => ({ ...prev, error: 'WebSocket is not connected' }));
      return;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      if (currentRequestIdRef.current && !message.request_id) {
        message = {
          ...message,
          request_id: currentRequestIdRef.current
        };
      }

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
  }, [autoConnect]);

  const clearMessages = useCallback(() => {
    currentRequestIdRef.current = null;
    setState(prev => ({ ...prev, messages: [] }));
  }, []);

  useEffect(() => {
    if (!autoConnect) return;

    connect();
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    state,
    sendMessage,
    connect,
    disconnect,
    clearMessages,
    getCurrentRequestId: () => currentRequestIdRef.current
  };
};
