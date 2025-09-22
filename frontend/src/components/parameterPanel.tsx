import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { InferenceRequest, InferenceMode, CompletedSequence, TokenSelection } from '../types/types';
import { DebugPanel } from './debugPanel';
import { HelpModal } from './helpModal';
import { TokenStreamPanel } from './tokenStreamPanel';

const Container = styled.div`
  display: grid;
  grid-template-columns: 30% 70%;
  gap: 16px;
  height: 100%;
  overflow: hidden;
  position: relative;
  box-sizing: border-box;
`;

const LeftSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  height: 100%;
  min-height: 0;
`;

const RightSection = styled.div`
  display: grid;
  grid-template-rows: 2fr 1fr;
  gap: 16px;
  min-height: 0;
`;

const RightTopSection = styled.div`
  padding: 16px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  display: flex;
  gap: 16px;
  align-items: stretch;
  min-height: 0;
`;

const SamplingParametersGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 60px;
  row-gap: 8px;
  max-width: 95%;
`;

const SectionHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e9ecef;
`;

const TitleSection = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: nowrap;
  overflow: hidden;
`;

const Title = styled.h1`
  margin: 0;
  color: #343a40;
  font-size: 24px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ConnectionStatus = styled.div<{ connected: boolean }>`
  padding: 4px 8px;
  background-color: ${props => props.connected ? '#d4edda' : '#f8d7da'};
  color: ${props => props.connected ? '#155724' : '#721c24'};
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
  display: flex;
  align-items: center;
  
  &::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: ${props => props.connected ? '#28a745' : '#dc3545'};
    margin-right: 6px;
  }
`;

const DebugButton = styled.button`
  background: #6c757d;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  
  &:hover {
    background: #5a6268;
  }
`;

const HelpButton = styled.button`
  background: #17a2b8;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  
  &:hover {
    background: #138496;
  }
`;

const CoverageButton = styled.button`
  background: #6f42c1;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  
  &:hover {
    background: #5a32a3;
  }
`;

const ParameterGroup = styled.div`
  margin-bottom: 8px;
  background: #f8f9fa;
  padding: 12px;
  border-radius: 6px;
  border: 1px solid #e9ecef;
  
  &.flex-fill {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    margin-bottom: 0;
  }
`;

const GroupTitle = styled.h3`
  margin: 0 0 8px 0;
  color: #495057;
  font-size: 14px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const HelperText = styled.p`
  margin: 8px 0 0 0;
  color: #6c757d;
  font-size: 12px;
  line-height: 1.4;
`;

const InputGroup = styled.div`
  margin-bottom: 8px;
  &:last-child {
    margin-bottom: 0;
  }
`;

const Label = styled.label`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
  color: #495057;
  font-weight: 500;
  font-size: 12px;
`;

const Tooltip = styled.span`
  color: #6c757d;
  font-size: 11px;
  font-weight: normal;
`;

const Input = styled.input`
  width: 100%;
  padding: 4px 6px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 13px;
  
  &:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0,123,255,0.15);
  }
`;

const Select = styled.select`
  width: 100%;
  padding: 6px 8px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 13px;
  background-color: white;
  cursor: pointer;
  
  &:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0,123,255,0.15);
  }
`;

const PromptContainer = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  flex: 1;
`;

const TextArea = styled.textarea`
  width: 100%;
  flex: 1;
  min-height: 50px;
  padding: 8px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 13px;
  resize: vertical;
  box-sizing: border-box;
  
  &:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0,123,255,0.15);
  }
  
  &::placeholder {
    color: #adb5bd;
  }
`;

const ExampleButtons = styled.div`
  position: absolute;
  top: -36px;
  right: 0;
  display: flex;
  gap: 4px;
`;

const ExampleButton = styled.button`
  padding: 6px 12px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  background: white;
  color: #495057;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: #f8f9fa;
    border-color: #007bff;
    color: #007bff;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  &:active {
    transform: translateY(0);
    box-shadow: none;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 16px;
  
  &.full-height {
    margin-top: 0;
    height: 100%;
    justify-content: space-between;
  }
`;

const Button = styled.button<{ variant?: 'primary' | 'secondary' | 'danger' }>`
  padding: 10px 20px;
  border-radius: 6px;
  border: none;
  background-color: ${props => {
    switch (props.variant) {
      case 'primary': return '#007bff';
      case 'danger': return '#dc3545';
      default: return '#6c757d';
    }
  }};
  color: white;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  
  .full-height & {
    flex: 1;
  }
  
  &:hover {
    background-color: ${props => {
      switch (props.variant) {
        case 'primary': return '#0056b3';
        case 'danger': return '#c82333';
        default: return '#5a6268';
      }
    }};
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  &:active {
    transform: translateY(0);
    box-shadow: none;
  }
  
  &:disabled {
    background-color: #adb5bd;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const ModelStatusBar = styled.div<{ status: 'unloaded' | 'loading' | 'loaded' | 'error' }>`
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 12px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  
  background-color: ${props => {
    switch (props.status) {
      case 'loading': return '#fff3cd';
      case 'loaded': return '#d4edda';
      case 'error': return '#f8d7da';
      default: return '#e9ecef';
    }
  }};
  
  color: ${props => {
    switch (props.status) {
      case 'loading': return '#856404';
      case 'loaded': return '#155724';
      case 'error': return '#721c24';
      default: return '#495057';
    }
  }};
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid #f3f3f3;
  border-top: 2px solid #856404;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ProgressBar = styled.div<{ progress: number }>`
  flex: 1;
  height: 8px;
  background-color: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
  
  &::after {
    content: '';
    display: block;
    width: ${props => props.progress}%;
    height: 100%;
    background-color: #007bff;
    transition: width 0.3s ease;
  }
`;

const WarningMessage = styled.div`
  padding: 12px 16px;
  background-color: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 6px;
  color: #856404;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 8px;
  
  svg {
    flex-shrink: 0;
  }
`;

const ButtonRow = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
`;

const PopupOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const PopupContent = styled.div`
  background: white;
  border-radius: 8px;
  padding: 24px;
  max-width: 400px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
`;

const PopupTitle = styled.h3`
  margin: 0 0 16px 0;
  color: #343a40;
  font-size: 18px;
  font-weight: 600;
`;

const PopupMessage = styled.p`
  margin: 0 0 24px 0;
  color: #495057;
  font-size: 14px;
  line-height: 1.5;
`;

const PopupButtons = styled.div`
  display: flex;
  gap: 12px;
  justify-content: flex-end;
`;

const PopupButton = styled.button<{ variant?: 'primary' | 'secondary' }>`
  padding: 8px 16px;
  border-radius: 4px;
  border: ${props => props.variant === 'primary' ? 'none' : '1px solid #ced4da'};
  background-color: ${props => props.variant === 'primary' ? '#007bff' : 'white'};
  color: ${props => props.variant === 'primary' ? 'white' : '#495057'};
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background-color: ${props => props.variant === 'primary' ? '#0056b3' : '#f8f9fa'};
    ${props => props.variant !== 'primary' && 'border-color: #adb5bd;'}
  }
`;

interface ParameterPanelProps {
  connected: boolean;
  error: string | null;
  messages: Array<{
    direction: 'sent' | 'received';
    timestamp: number;
    data: any;
  }>;
  onSendMessage: (message: InferenceRequest) => void;
  onConnect: () => void;
  onDisconnect: () => void;
  onClearMessages: () => void;
  onRequestTokenGeneration?: React.MutableRefObject<(
    nodeId: string, 
    requestId?: string
  ) => Promise<boolean>>;
  modelStatus: 'unloaded' | 'loading' | 'loaded' | 'error';
  loadingProgress: number;
  statusMessage: string;
  isGenerating: boolean;
  hasTree?: boolean;
  onUpdateParameters?: (params: {
    k: number;
    temperature: number;
    topP: number;
    minP: number;
    maxTokens: number;
  }) => void;
  
  completedSequence: CompletedSequence | null;
  selectedToken: TokenSelection;
  onTokenSelect: (sequenceId: string, tokenIndex: number) => void;
  alternatives?: Array<{ token: string; probability: number; nodeId: string }>;
  completedSequences?: CompletedSequence[];
  onSequenceSelect?: (sequence: CompletedSequence) => void;
}

const DEFAULT_PROMPT = "How many r in strawberry?";

const EXAMPLE_PROMPTS = [
  "Give me the very short, one sentence poem. Return the sentence only.",
  "What is 9.9 - 9.11?",
  "How many r in strawberry?",
   "Give me the random number, in 10 digits.",
];

export const ParameterPanel: React.FC<ParameterPanelProps> = ({
  connected,
  error,
  messages,
  onSendMessage,
  onConnect,
  onDisconnect,
  onClearMessages,
  onRequestTokenGeneration,
  modelStatus,
  loadingProgress,
  statusMessage,
  isGenerating,
  hasTree = false,
  onUpdateParameters,
  
  completedSequence,
  selectedToken,
  onTokenSelect,
  alternatives = [],
  completedSequences = [],
  onSequenceSelect
}) => {
  const [inputText, setInputText] = useState(DEFAULT_PROMPT);
  const [mode, setMode] = useState<InferenceMode>('generate_with_smc');
  const [k, setK] = useState(5);
  const [maxTokens, setMaxTokens] = useState(20);
  const [treeDepth, setTreeDepth] = useState(3);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [minP, setMinP] = useState(0.05);
  
  
  useEffect(() => {
    if (onUpdateParameters) {
      onUpdateParameters({
        k,
        temperature,
        topP,
        minP,
        maxTokens
      });
    }
  }, [k, temperature, topP, minP, maxTokens, onUpdateParameters]);
  
  
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  
  const [isDebugOpen, setIsDebugOpen] = useState(false);
  
  
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isCoverageOpen, setIsCoverageOpen] = useState(false);

  
  const [currentRequestId, setCurrentRequestId] = useState<string | null>(null);
  
  
  const [initialPrompt, setInitialPrompt] = useState(DEFAULT_PROMPT);
  const promptChanged = inputText !== initialPrompt && hasTree;
  
  
  const [showConfirmPopup, setShowConfirmPopup] = useState(false);
  
  
  const [isFileSelectOpen, setIsFileSelectOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [localFile, setLocalFile] = useState<File | null>(null);
  
  
  useEffect(() => {
    const hasVisited = localStorage.getItem('inferactive_help_dismissed');
    if (!hasVisited) {
      setIsHelpOpen(true);
    }
  }, []);
  
  
  const handleTokenGenerationRequest = async (
    nodeId: string
  ): Promise<boolean> => {
    
    if (!connected) {
      console.warn('WebSocket not connected.');
      return false;
    }
    
    
    const message: InferenceRequest = {
      type: 'explore_node',
      input_text: inputText,
      k,
      temperature,
      top_p: topP,
      min_p: minP,
      max_tokens: maxTokens,
      node_id: nodeId
    };
  
    
    try {
      onSendMessage(message);
      return true;
    } catch (error) {
      console.error('error in sending:', error);
      return false;
    }
  };
  
  
  useEffect(() => {
    if (onRequestTokenGeneration) {
      onRequestTokenGeneration.current = handleTokenGenerationRequest;
    }
  }, [onRequestTokenGeneration, connected, inputText, k, temperature, topP, minP, maxTokens, currentRequestId]);

  
  const handleSend = () => {
    
    if (hasTree) {
      setShowConfirmPopup(true);
      return;
    }
    
    
    setIsSettingsOpen(false);
    
    
    const message: InferenceRequest = {
      type: mode,
      input_text: inputText,
      k,
      temperature,
      top_p: topP,
      min_p: minP,
      ...(mode === 'generate_with_smc'
        ? { max_tokens: maxTokens } 
        : { depth: treeDepth })
    };
    
    
    onSendMessage(message);
    
    setInitialPrompt(inputText);
  };
  
  
  const handleConfirmGenerate = () => {
    setShowConfirmPopup(false);
    
    
    onClearMessages();
    
    
    setTimeout(() => {
      setIsSettingsOpen(false);
      
      const message: InferenceRequest = {
        type: mode,
        input_text: inputText,
        k,
        temperature,
        top_p: topP,
        min_p: minP,
        ...(mode === 'generate_with_smc'
          ? { max_tokens: maxTokens } 
          : { depth: treeDepth })
      };
      
      onSendMessage(message);
      setInitialPrompt(inputText);
    }, 100);
  };
  
  const handleClear = () => {
    onClearMessages();
    
    setInitialPrompt(inputText);
  };
  
  
  const publicJsonFiles = [
    'prompt-1-A.json',
    'prompt-1-B.json',
    'prompt-2-A.json', 
    'prompt-2-B.json',
  ];

  
  const processFileData = (data: any) => {
    const treeData = data.tree || data;

    if (data.metadata?.prompt) {
      setInputText(data.metadata.prompt);
      setInitialPrompt(data.metadata.prompt);
    }

    const fakeWebSocketMessage = {
      type: 'tree_result',
      request_id: 'file_load_' + Date.now(),
      tree: treeData
    };

    onClearMessages();
    (window as any).__loadedTreeData = fakeWebSocketMessage;
    window.dispatchEvent(new CustomEvent('treeFileLoaded'));
    setIsFileSelectOpen(false);
  };

  const handleFileLoad = async (filename: string) => {
    try {
      const response = await fetch(`/${filename}`);
      if (!response.ok) throw new Error(`Failed to load ${filename}`);

      const data = await response.json();
      processFileData(data);
    } catch (error) {
      console.error('Failed to load JSON file:', error);
      alert('Failed to load JSON file');
    }
  };

  const handleLocalFileLoad = async (file: File) => {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      processFileData(data);
    } catch (error) {
      console.error('Failed to load local JSON file:', error);
      alert('Failed to load local JSON file');
    }
  };

  
  const DebugIcon = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
      <line x1="6" y1="6" x2="6.01" y2="6"></line>
      <line x1="6" y1="18" x2="6.01" y2="18"></line>
    </svg>
  );
  
  
  const HelpIcon = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"></circle>
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
      <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>
  );

  const ChevronIcon = ({ isUp }: { isUp: boolean }) => (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points={isUp ? "18 15 12 9 6 15" : "6 9 12 15 18 9"} />
    </svg>
  );
  
  const TreeIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
      <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
      <line x1="12" y1="22.08" x2="12" y2="12"></line>
    </svg>
  );
  
  const ClearIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18"></path>
      <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path>
      <line x1="10" y1="11" x2="10" y2="17"></line>
      <line x1="14" y1="11" x2="14" y2="17"></line>
    </svg>
  );
  
  
  const FileIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="12" y1="18" x2="12" y2="12"/>
      <line x1="9" y1="15" x2="15" y2="15"/>
    </svg>
  );
  
  const AlertIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"></circle>
      <line x1="12" y1="8" x2="12" y2="12"></line>
      <line x1="12" y1="16" x2="12.01" y2="16"></line>
    </svg>
  );

  return (
    <Container>
      <LeftSection>
        <SectionHeader>
          <TitleSection>
            <Title>InFerActive</Title>
            <ConnectionStatus connected={connected} title={connected ? "connecting to server" : "no connect"}>
              {connected ? 'Connected' : 'Disconnected'}
            </ConnectionStatus>
          </TitleSection>
          <ButtonRow>
            <HelpButton onClick={() => setIsHelpOpen(true)}>
              <HelpIcon />
              Help
            </HelpButton>
            <DebugButton onClick={() => setIsDebugOpen(!isDebugOpen)}>
              <DebugIcon />
              Debug
            </DebugButton>
          </ButtonRow>
        </SectionHeader>
        
        <ModelStatusBar status={modelStatus}>
          {modelStatus === 'loading' && (
            <>
              <LoadingSpinner />
              <span>Loading model... {loadingProgress}%</span>
              {loadingProgress > 0 && <ProgressBar progress={loadingProgress} />}
            </>
          )}
          {modelStatus === 'loaded' && !isGenerating && (
            <span>✓ Model loaded</span>
          )}
          {modelStatus === 'loaded' && isGenerating && (
            <>
              <LoadingSpinner />
              <span>Generating response...</span>
            </>
          )}
          {modelStatus === 'error' && (
            <span>✗ Error: {statusMessage}</span>
          )}
          {modelStatus === 'unloaded' && (
            <span>Model not loaded (will load on first inference)</span>
          )}
        </ModelStatusBar>
        
        <ParameterGroup>
          <GroupTitle>Sampling Parameters</GroupTitle>
          <SamplingParametersGrid>
            <InputGroup>
              <Label>
                Top-K
                <Tooltip>Number of tokens to consider</Tooltip>
              </Label>
              <Input
                type="number"
                value={k}
                onChange={(e) => setK(parseInt(e.target.value))}
                min={1}
              />
            </InputGroup>
            
            <InputGroup>
              <Label>
                Temperature
                <Tooltip>Randomness (0.0-2.0)</Tooltip>
              </Label>
              <Input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                step="0.1"
                min={0}
                max={2}
              />
            </InputGroup>
            
            <InputGroup>
              <Label>
                Top-P
                <Tooltip>Nucleus sampling threshold</Tooltip>
              </Label>
              <Input
                type="number"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                step="0.1"
                min={0}
                max={1}
              />
            </InputGroup>

            <InputGroup>
              <Label>
                Min-P
                <Tooltip>Minimum probability threshold</Tooltip>
              </Label>
              <Input
                type="number"
                value={minP}
                onChange={(e) => setMinP(parseFloat(e.target.value))}
                step="0.01"
                min={0}
                max={1}
              />
            </InputGroup>
          </SamplingParametersGrid>
        </ParameterGroup>
      </LeftSection>
      
      <RightSection>
        <RightTopSection>
          <ParameterGroup style={{ flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100% - 8px)', marginBottom: '8px' }}>
            <GroupTitle>Prompt</GroupTitle>
            <PromptContainer>
              <ExampleButtons>
                {[1, 2, 3, 4].map((num) => (
                  <ExampleButton
                    key={num}
                    onClick={() => setInputText(EXAMPLE_PROMPTS[num - 1])}
                    title={EXAMPLE_PROMPTS[num - 1]}
                  >
                    {num}
                  </ExampleButton>
                ))}
              </ExampleButtons>
              <TextArea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter your prompt here..."
              />
            </PromptContainer>
          </ParameterGroup>

          <ButtonGroup className="full-height" style={{ flexShrink: 0 }}>
            <Button 
              variant="primary" 
              onClick={handleSend} 
              disabled={!connected || !inputText || modelStatus === 'loading' || isGenerating}
            >
              <TreeIcon />
              {modelStatus === 'loading' ? 'Loading Model...' : 
               isGenerating ? 'Generating Tree...' : 'Generate New Tree'}
            </Button>
            
            <Button
              variant="secondary"
              onClick={() => setIsFileSelectOpen(true)}
              disabled={isGenerating}
            >
              <FileIcon />
              Load from File
            </Button>
          </ButtonGroup>
        </RightTopSection>
        
        <TokenStreamPanel
          completedSequence={completedSequence}
          selectedToken={selectedToken}
          onTokenSelect={onTokenSelect}
          alternatives={alternatives}
          completedSequences={completedSequences}
          onSequenceSelect={onSequenceSelect}
        />
      </RightSection>

      {isDebugOpen && (
        <DebugPanel
          connected={connected}
          messages={messages}
          onClose={() => setIsDebugOpen(false)}
        />
      )}
      
      <HelpModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
      />
      
      {isFileSelectOpen && (
        <PopupOverlay onClick={() => setIsFileSelectOpen(false)}>
          <PopupContent onClick={(e) => e.stopPropagation()}>
            <PopupTitle>Load JSON File</PopupTitle>
            <PopupMessage>
              Select from public directory or upload local file:
            </PopupMessage>
            <Select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              style={{ marginBottom: '8px' }}
            >
              <option value="">Select public JSON file...</option>
              {publicJsonFiles.map(file => (
                <option key={file} value={file}>{file}</option>
              ))}
            </Select>
            <div style={{ textAlign: 'center', margin: '8px 0', color: '#6c757d', fontSize: '12px' }}>OR</div>
            <input
              type="file"
              accept=".json"
              onChange={(e) => {
                setLocalFile(e.target.files?.[0] || null);
                setSelectedFile('');
              }}
              style={{ marginBottom: '16px', width: '100%' }}
            />
            <PopupButtons>
              <PopupButton variant="secondary" onClick={() => setIsFileSelectOpen(false)}>
                Cancel
              </PopupButton>
              <PopupButton
                variant="primary"
                onClick={() => {
                  if (selectedFile) {
                    handleFileLoad(selectedFile);
                  } else if (localFile) {
                    handleLocalFileLoad(localFile);
                  }
                }}
                disabled={!selectedFile && !localFile}
              >
                Load
              </PopupButton>
            </PopupButtons>
          </PopupContent>
        </PopupOverlay>
      )}

      {showConfirmPopup && (
        <PopupOverlay onClick={() => setShowConfirmPopup(false)}>
          <PopupContent onClick={(e) => e.stopPropagation()}>
            <PopupTitle>Clear Current Tree?</PopupTitle>
            <PopupMessage>
              A tree already exists. This action will clear the current tree and generate a new one. Do you want to continue?
            </PopupMessage>
            <PopupButtons>
              <PopupButton variant="secondary" onClick={() => setShowConfirmPopup(false)}>
                No
              </PopupButton>
              <PopupButton variant="primary" onClick={handleConfirmGenerate}>
                Yes
              </PopupButton>
            </PopupButtons>
          </PopupContent>
        </PopupOverlay>
      )}
    </Container>
  );
};