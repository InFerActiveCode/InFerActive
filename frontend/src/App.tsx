import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { TokenTreeVisualizer } from './components/tokenTreePanel';
import { TokenStreamPanel } from './components/tokenStreamPanel';
import { useWebSocket } from './hooks/useWebSocket';
import { useTreeManager } from './hooks/useTreeManager';
import { TokenGenerationService } from './services/tokenGenService';
import { ApiNode, CompletedSequence } from './types/types';
import { collectLeafSequences, findNodeById, generateSequenceFromNode } from './utils/treeTransform';
import { createLeafLimitedTree } from './utils/leafLimitTree';

const MIN_OVERVIEW_DEPTH = 1;
const MAX_OVERVIEW_DEPTH = 20;
const DEFAULT_OVERVIEW_DEPTH = 3;
const MIN_FILTERING_MIN_PROB = 0;
const DEFAULT_FILTERING_MIN_PROB = 0.01;
const MAX_FILTERING_MIN_PROB = 1;
const MIN_FILTERING_MAX_BRANCHES = 1;
const MAX_FILTERING_MAX_BRANCHES = 10;
const DEFAULT_FILTERING_MAX_BRANCHES = 3;
const FORCE_MOCK_CONNECTED_UI = process.env.REACT_APP_FORCE_MOCK_CONNECTED_UI === 'true';
const MOCK_BACKEND_TREE_PATH = '/prompt-1-A.json';

type JsonObject = Record<string, unknown>;
type TreeJsonNode = Omit<ApiNode, 'children'> & {
  children: TreeJsonNode[];
  continuation?: string;
};

const AppContainer = styled.div`
  height: 100vh;
  gap: var(--genesis-space-4);
  padding: var(--genesis-space-4);
  background: var(--genesis-background);
  overflow: hidden;
  max-width: 1920px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
`;

const TopBar = styled.div`
  display: grid;
  grid-template-columns: minmax(320px, 400px) minmax(260px, 320px) minmax(360px, 1fr);
  gap: var(--genesis-space-3);
  flex-shrink: 0;
  align-items: start;

  @media (max-width: 1180px) {
    grid-template-columns: 1fr;
    overflow: auto;
    max-height: 45vh;
  }
`;

const ControlCard = styled.section`
  background: var(--genesis-surface);
  border-radius: var(--genesis-radius-md);
  border: 1px solid var(--genesis-border);
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 0;
  overflow: hidden;
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--genesis-space-3);
`;

const CardTitle = styled.h2`
  margin: 0;
  font-size: 14px;
  font-weight: 700;
  color: var(--genesis-text-primary);
`;

const StatusPill = styled.span<{ $tone: 'live' | 'idle' | 'warn' }>`
  flex-shrink: 0;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  color: ${p => {
    if (p.$tone === 'live') return '#0f5132';
    if (p.$tone === 'warn') return '#842029';
    return '#475467';
  }};
  background: ${p => {
    if (p.$tone === 'live') return '#d1e7dd';
    if (p.$tone === 'warn') return '#f8d7da';
    return '#e9eef5';
  }};
`;

const ParameterHeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: var(--genesis-space-2);
  justify-content: flex-end;
`;

const ParameterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--genesis-space-2) var(--genesis-space-3);
`;

const AdditionalParametersRegion = styled.div`
  max-height: 120px;
  overflow: hidden;
  animation: app-additional-parameters-open 0.18s ease;

  @keyframes app-additional-parameters-open {
    from {
      max-height: 0;
      opacity: 0;
    }
    to {
      max-height: 120px;
      opacity: 1;
    }
  }
`;

const AdditionalParameterGrid = styled(ParameterGrid)`
  padding-top: var(--genesis-space-3);
  border-top: 1px solid var(--genesis-border);
`;

const ParameterField = styled.label`
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
`;

const ParameterLabelRow = styled.div`
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
`;

const ParameterLabel = styled.span`
  font-size: 12px;
  font-weight: 700;
  color: var(--genesis-text-primary);
`;

const ParameterInput = styled.input`
  width: 100%;
  padding: 9px 10px;
  border: 1px solid var(--genesis-border-input);
  border-radius: var(--genesis-radius-sm);
  background: var(--genesis-surface);
  color: var(--genesis-text-primary);
  font-size: 13px;

  &:focus {
    outline: none;
    border-color: var(--genesis-primary);
    box-shadow: var(--genesis-focus-ring);
    background: var(--genesis-surface);
  }
`;

const BackendGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 6px;
`;

const BackendStat = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 6px 9px;
  border-radius: var(--genesis-radius-sm);
  border: 1px solid var(--genesis-border);
  background: var(--genesis-background);
  color: var(--genesis-text-secondary);
  font-size: 12px;
`;

const BackendStatValue = styled.span`
  color: var(--genesis-text-primary);
  font-weight: 700;
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const ActionRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;

const ActionButton = styled.button<{ $secondary?: boolean }>`
  padding: 8px 12px;
  border-radius: var(--genesis-radius-sm);
  border: 1px solid ${p => (p.$secondary ? 'var(--genesis-border-input)' : 'var(--genesis-primary)')};
  background: ${p => (p.$secondary ? '#f8fafc' : 'var(--genesis-primary)')};
  color: ${p => (p.$secondary ? 'var(--genesis-text-primary)' : '#fff')};
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;

  &:hover:not(:disabled) {
    background: ${p => (p.$secondary ? '#f1f5f9' : 'var(--genesis-primary-hover)')};
    border-color: ${p => (p.$secondary ? '#b8c3d3' : 'var(--genesis-primary-hover)')};
  }

  &:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--genesis-focus-ring);
  }
`;

const BackendCard = styled(ControlCard)`
  padding: 10px 14px;
  gap: 8px;
`;

const HiddenFileInput = styled.input`
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
`;

const PromptCard = styled(ControlCard)`
  gap: 10px;
`;

const PromptHeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: var(--genesis-space-2);
  flex-wrap: wrap;
  justify-content: flex-end;
`;

const SourceBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--genesis-background);
  border: 1px solid var(--genesis-border);
  color: var(--genesis-text-secondary);
  font-size: 11px;
  font-weight: 600;
  font-family: source-code-pro, Menlo, Monaco, Consolas, monospace;
`;

const FileUploadLabel = styled.label`
  padding: 8px 12px;
  border-radius: var(--genesis-radius-sm);
  border: 1px solid var(--genesis-border-input);
  background: #f8fafc;
  color: var(--genesis-text-primary);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
  white-space: nowrap;

  &:hover {
    background: #f1f5f9;
    border-color: #b8c3d3;
  }

  &:focus-within {
    outline: none;
    box-shadow: var(--genesis-focus-ring);
  }
`;

const PromptTextArea = styled.textarea`
  flex: 1;
  width: 100%;
  min-height: 88px;
  resize: none;
  padding: 12px 14px;
  border: 1px solid var(--genesis-border-input);
  border-radius: var(--genesis-radius-md);
  background: var(--genesis-surface);
  color: var(--genesis-text-primary);
  font-size: 14px;
  line-height: 1.6;

  &:focus {
    outline: none;
    border-color: var(--genesis-primary);
    box-shadow: var(--genesis-focus-ring);
    background: var(--genesis-surface);
  }

  &::placeholder {
    color: var(--genesis-neutral);
  }
`;

const InlineNote = styled.div`
  flex: 1 1 260px;
  min-width: 220px;
  min-height: 32px;
  color: var(--genesis-text-secondary);
  font-size: 12px;
  line-height: 1.35;
  display: flex;
  align-items: center;
`;

const BottomArea = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 1fr) 420px;
  gap: var(--genesis-space-3);
  flex: 1;
  min-height: 0;
  overflow: hidden;

  @media (max-width: 1180px) {
    grid-template-columns: 1fr;
  }
`;

const MainContentArea = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
`;

const ContentContainer = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: hidden;
`;

const TreeToolbar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--genesis-space-3);
  padding: 8px 12px;
  background: var(--genesis-surface);
  border: 1px solid var(--genesis-border);
  border-bottom: none;
  border-radius: var(--genesis-radius-md) var(--genesis-radius-md) 0 0;
  user-select: none;
  flex-wrap: wrap;
`;

const TreeToolbarMain = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1px;
`;

const TreeToolbarTitle = styled.div`
  font-size: 14px;
  font-weight: 700;
  color: var(--genesis-text-primary);
`;

const TreeToolbarMeta = styled.div`
  font-size: 12px;
  color: var(--genesis-text-secondary);
`;

const ToolbarActions = styled.div`
  display: flex;
  align-items: center;
  align-self: flex-start;
  gap: 6px;
  flex-wrap: wrap;
`;

const ToolbarButton = styled.button<{ $active: boolean; $tone?: 'neutral' | 'good' | 'bad' }>`
  min-width: 34px;
  padding: 4px 9px;
  font-size: 11px;
  font-weight: 700;
  line-height: 16px;
  border-radius: 7px;
  border: 1px solid ${p => {
    if (p.$active && p.$tone === 'good') return 'var(--genesis-success)';
    if (p.$active && p.$tone === 'bad') return 'var(--genesis-error)';
    if (p.$active) return '#607d8b';
    return 'var(--genesis-border-input)';
  }};
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
  background: ${p => {
    if (p.$active && p.$tone === 'good') return 'var(--genesis-success)';
    if (p.$active && p.$tone === 'bad') return 'var(--genesis-error)';
    if (p.$active) return '#607d8b';
    return 'var(--genesis-background)';
  }};
  color: ${p => (p.$active ? '#fff' : 'var(--genesis-text-secondary)')};

  &:hover {
    border-color: ${p => {
      if (p.$tone === 'good') return 'var(--genesis-success)';
      if (p.$tone === 'bad') return 'var(--genesis-error)';
      return '#607d8b';
    }};
    color: ${p => (p.$active ? '#fff' : 'var(--genesis-text-primary)')};
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--genesis-focus-ring);
  }
`;

const TreeVisualizerContainer = styled.div`
  position: relative;
  border: 1px solid var(--genesis-border);
  border-top: none;
  border-radius: 0 0 var(--genesis-radius-md) var(--genesis-radius-md);
  padding: var(--genesis-space-4);
  background: var(--genesis-surface);
  overflow: auto;
  flex: 1;
  min-height: 0;
`;

const TreePanelControls = styled.div`
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 1100;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
`;

const TreeSettingsAnchor = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
`;

const IconToolbarButton = styled.button<{ $active?: boolean }>`
  width: 46px;
  height: 46px;
  padding: 0;
  border: 1px solid ${p => (p.$active ? '#607d8b' : 'var(--genesis-border-input)')};
  border-radius: 8px;
  background: ${p => (p.$active ? '#607d8b' : 'rgba(255, 255, 255, 0.96)')};
  color: ${p => (p.$active ? '#fff' : 'var(--genesis-text-secondary)')};
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s, border-color 0.15s, box-shadow 0.15s, color 0.15s;

  svg {
    width: 22px;
    height: 22px;
  }

  &:hover {
    border-color: #607d8b;
    color: ${p => (p.$active ? '#fff' : 'var(--genesis-text-primary)')};
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--genesis-focus-ring);
  }
`;

const TreeSettingsPanel = styled.div`
  position: absolute;
  top: 0;
  right: calc(100% + 8px);
  z-index: 20;
  width: 270px;
  padding: 12px;
  border: 1px solid var(--genesis-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const SettingsSliderRow = styled.div`
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr) 24px 48px;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--genesis-text-secondary);
  min-width: 0;
`;

const SettingsSliderLabel = styled.span`
  grid-column: 1 / -1;
  font-weight: 700;
  color: var(--genesis-text-primary);
  white-space: nowrap;
`;

const SettingsSliderInput = styled.input`
  width: 100%;
  min-width: 0;
`;

const SettingsSliderValue = styled.span`
  text-align: right;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  color: var(--genesis-text-secondary);
`;

const SettingsStepButton = styled.button`
  width: 24px;
  height: 24px;
  padding: 0;
  border: 1px solid var(--genesis-border-input);
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.96);
  color: var(--genesis-text-primary);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;

  &:hover {
    border-color: #607d8b;
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--genesis-focus-ring);
  }
`;

const EmptyState = styled.div`
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--genesis-text-secondary);
  padding: 40px;
  font-size: 14px;
`;

const StreamCard = styled.div`
  min-height: 0;
  overflow: hidden;
`;

const GlobalLoadingOverlay = styled.div`
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(20, 25, 32, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
`;

const GlobalLoadingCard = styled.div`
  background: var(--genesis-surface);
  border: 1px solid var(--genesis-border);
  border-radius: 10px;
  padding: 16px 20px;
  min-width: 240px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
  display: flex;
  align-items: center;
  gap: var(--genesis-space-3);
  color: var(--genesis-text-primary);
  font-size: 14px;
  font-weight: 600;
`;

const LoadingSpinner = styled.div`
  width: 18px;
  height: 18px;
  border: 2px solid #cfd8e3;
  border-top-color: var(--genesis-primary);
  border-radius: 50%;
  animation: app-loading-spin 0.9s linear infinite;

  @keyframes app-loading-spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const isJsonObject = (value: unknown): value is JsonObject => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

const stringValue = (value: unknown): string | undefined => (
  typeof value === 'string' ? value : undefined
);

const nestedObject = (data: JsonObject, key: string): JsonObject | null => {
  const value = data[key];
  return isJsonObject(value) ? value : null;
};

const applyContinuation = (node: TreeJsonNode): TreeJsonNode => {
  const nextNode: TreeJsonNode = { ...node };
  if (node.continuation) {
    nextNode.text = `${node.text || ''}${node.continuation}`;
  }
  nextNode.children = Array.isArray(node.children)
    ? node.children.map(applyContinuation)
    : [];
  return nextNode;
};

const extractTreeData = (data: unknown): TreeJsonNode | null => {
  if (!isJsonObject(data)) return null;

  const dataObject = nestedObject(data, 'data');
  const resultObject = nestedObject(data, 'result');
  const responseObject = nestedObject(data, 'response');
  const treeResultObject = nestedObject(data, 'tree_result');
  const treeResultCamelObject = nestedObject(data, 'treeResult');
  const candidates = [
    data.tree,
    dataObject?.tree,
    resultObject?.tree,
    responseObject?.tree,
    treeResultObject?.tree,
    treeResultCamelObject?.tree,
  ];

  for (const candidate of candidates) {
    if (isJsonObject(candidate)) return candidate as TreeJsonNode;
  }

  if ('id' in data || Array.isArray(data.children)) {
    return data as TreeJsonNode;
  }
  return null;
};

const extractPromptText = (data: unknown, fallback: string): string => {
  if (!isJsonObject(data)) return fallback;

  const metadata = nestedObject(data, 'metadata');
  const request = nestedObject(data, 'request');
  return (
    stringValue(metadata?.prompt) ??
    stringValue(metadata?.input_text) ??
    stringValue(data.prompt) ??
    stringValue(data.input_text) ??
    stringValue(request?.input_text) ??
    fallback
  );
};

const clampSettingValue = (value: number, min: number, max: number): number => (
  Math.min(max, Math.max(min, value))
);

const formatPercent = (value: number): string => `${value.toFixed(2)}%`;

const GearIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M9.59 3.94A1.12 1.12 0 0 1 10.7 3h2.6c.55 0 1.02.4 1.11.94l.21 1.28c.06.37.31.69.64.87l.22.13c.33.2.72.26 1.08.12l1.22-.45c.51-.19 1.09.01 1.37.49l1.3 2.25c.27.48.16 1.08-.26 1.43l-1 .83c-.29.24-.44.61-.43.99v.26c-.01.38.14.75.43.99l1 .83c.42.35.53.95.26 1.43l-1.3 2.25c-.28.48-.86.68-1.37.49l-1.22-.46c-.36-.13-.75-.07-1.08.12l-.22.13c-.33.18-.58.5-.64.87l-.21 1.28c-.09.54-.56.94-1.11.94h-2.6c-.55 0-1.02-.4-1.11-.94l-.21-1.28c-.06-.37-.31-.69-.64-.87l-.22-.13c-.33-.19-.72-.25-1.08-.12l-1.22.46c-.51.19-1.09-.01-1.37-.49l-1.3-2.25a1.12 1.12 0 0 1 .26-1.43l1-.83c.29-.24.44-.61.43-.99v-.26c.01-.38-.14-.75-.43-.99l-1-.83a1.12 1.12 0 0 1-.26-1.43l1.3-2.25c.28-.48.86-.68 1.37-.49l1.22.45c.36.14.75.08 1.08-.12l.22-.13c.33-.18.58-.5.64-.87l.21-1.28z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const TreeModeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="5" cy="12" r="2" />
    <circle cx="18" cy="6" r="2" />
    <circle cx="18" cy="18" r="2" />
    <path d="M7 12h5" />
    <path d="M12 12V6h4" />
    <path d="M12 12v6h4" />
  </svg>
);

interface SliderSettingProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  formatValue?: (value: number) => string;
}

const SliderSetting = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  formatValue = nextValue => `${nextValue}`,
}: SliderSettingProps) => {
  const updateValue = (nextValue: number) => {
    onChange(clampSettingValue(Number(nextValue.toFixed(4)), min, max));
  };

  return (
    <SettingsSliderRow>
      <SettingsSliderLabel>{label}</SettingsSliderLabel>
      <SettingsStepButton type="button" onClick={() => updateValue(value - step)} aria-label={`Decrease ${label}`}>
        -
      </SettingsStepButton>
      <SettingsSliderInput
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={event => updateValue(Number(event.target.value))}
      />
      <SettingsStepButton type="button" onClick={() => updateValue(value + step)} aria-label={`Increase ${label}`}>
        +
      </SettingsStepButton>
      <SettingsSliderValue>{formatValue(value)}</SettingsSliderValue>
    </SettingsSliderRow>
  );
};

function App() {
  const [tempSequence, setTempSequence] = useState<CompletedSequence | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isJsonFileLoading, setIsJsonFileLoading] = useState(false);
  const [isOverviewEnabled, setIsOverviewEnabled] = useState(false);
  const [overviewDepth, setOverviewDepth] = useState(DEFAULT_OVERVIEW_DEPTH);
  const [isTreeSettingsOpen, setIsTreeSettingsOpen] = useState(false);
  const [filteringMinProb, setFilteringMinProb] = useState(DEFAULT_FILTERING_MIN_PROB);
  const [filteringMaxBranches, setFilteringMaxBranches] = useState(DEFAULT_FILTERING_MAX_BRANCHES);
  const [promptText, setPromptText] = useState(TokenGenerationService.getDefaultParameters().inputText);
  const [panelResetKey, setPanelResetKey] = useState(0);
  const [evaluationColorEnabled, setEvaluationColorEnabled] = useState(true);
  const [leafLimitOverride, setLeafLimitOverride] = useState<number | null>(null);
  const [isAdditionalParametersOpen, setIsAdditionalParametersOpen] = useState(false);
  const [loadedSourceLabel, setLoadedSourceLabel] = useState('live backend');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    state: webSocketState,
    sendMessage,
    connect,
    disconnect,
    clearMessages,
  } = useWebSocket(!FORCE_MOCK_CONNECTED_UI);

  const {
    visualTree,
    completedSequences,
    selectedToken,
    parameters,
    modelStatus,
    loadingProgress,
    statusMessage,
    isGenerating,
    evaluatedNodeIds,
    evaluatedPathsTotal,
    evaluationFilters,
    resetAll,
    handleTokenSelect,
    handleNodePin,
    handleNodeEvaluate,
    toggleEvaluationFilter,
    handleRequestTokenGeneration,
    sendInferenceRequest,
    getSelectedSequence,
    isNodeGenerating,
    updateParameters,
    loadTreeFromData,
  } = useTreeManager({
    webSocketState,
    sendMessage,
    clearMessages,
    tempSequence,
  });

  useEffect(() => {
    updateParameters({ inputText: promptText });
  }, [promptText, updateParameters]);

  const displayVisualTree = useMemo(() => {
    if (!visualTree || leafLimitOverride === null) return visualTree;
    return createLeafLimitedTree(visualTree, leafLimitOverride);
  }, [visualTree, leafLimitOverride]);

  const displayCompletedSequences = useMemo(() => {
    if (leafLimitOverride === null) return completedSequences;
    if (!displayVisualTree) return [];
    return collectLeafSequences(displayVisualTree);
  }, [completedSequences, displayVisualTree, leafLimitOverride]);

  const displaySelectedSequence = useMemo(() => {
    if (tempSequence) return tempSequence;
    if (selectedToken.sequenceId) {
      const matchedDisplaySequence = displayCompletedSequences.find(
        sequence => sequence.id === selectedToken.sequenceId
      );
      if (matchedDisplaySequence) return matchedDisplaySequence;
    }
    return getSelectedSequence();
  }, [displayCompletedSequences, getSelectedSequence, selectedToken.sequenceId, tempSequence]);

  const resetViewerState = useCallback(() => {
    setTempSequence(null);
    setSelectedNodeId(null);
    setPanelResetKey(prev => prev + 1);
    resetAll();
  }, [resetAll]);

  const processLoadedData = useCallback((data: unknown, sourceLabel: string) => {
    const rawTreeData = extractTreeData(data);
    if (!rawTreeData) {
      throw new Error('JSON does not contain a token tree');
    }

    const treeData = applyContinuation(rawTreeData);
    const nextPromptText = extractPromptText(data, promptText);

    setLoadedSourceLabel(sourceLabel);
    setPromptText(nextPromptText);
    setEvaluationColorEnabled(true);
    setLeafLimitOverride(null);
    setTempSequence(null);
    setSelectedNodeId(null);
    setPanelResetKey(prev => prev + 1);
    loadTreeFromData(treeData);
  }, [loadTreeFromData, promptText]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsJsonFileLoading(true);
    const reader = new FileReader();
    reader.onload = (event) => {
      setTimeout(() => {
        try {
          const data = JSON.parse(event.target?.result as string);
          processLoadedData(data, file.name);
        } catch (error) {
          alert(error instanceof Error ? error.message : 'Invalid JSON file');
        } finally {
          setIsJsonFileLoading(false);
        }
      }, 0);
    };
    reader.onerror = () => {
      setIsJsonFileLoading(false);
      alert('Failed to read JSON file');
    };
    reader.readAsText(file);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [processLoadedData]);

  const handleTreeNodeSelect = useCallback((sequenceId: string, tokenIndex: number) => {
    if (sequenceId === '' && tokenIndex === -1) {
      handleTokenSelect('', -1);
      setTempSequence(null);
      setSelectedNodeId(null);
      return;
    }

    if (sequenceId.startsWith('temp-')) {
      const nodeId = sequenceId.replace('temp-', '');
      setSelectedNodeId(nodeId);
      if (displayVisualTree) {
        if (nodeId === displayVisualTree.id) {
          setTempSequence(null);
          handleTokenSelect(`temp-${nodeId}`, 0);
          return;
        }

        const generatedSequence = generateSequenceFromNode(displayVisualTree, nodeId);
        if (generatedSequence) {
          setTempSequence(generatedSequence);
          const actualTokenIndex = generatedSequence.tokens.findIndex(token => token.nodeId === nodeId);
          handleTokenSelect(generatedSequence.id, actualTokenIndex >= 0 ? actualTokenIndex : 0);
          return;
        }
      }
      return;
    }

    const isSelectingCurrentTemp = !!tempSequence && sequenceId === tempSequence.id;
    if (!isSelectingCurrentTemp) {
      setTempSequence(null);
    }
    setSelectedNodeId(null);
    handleTokenSelect(sequenceId, tokenIndex);
  }, [displayVisualTree, handleTokenSelect, tempSequence]);

  const handleParameterChange = useCallback((
    key: 'k' | 'temperature' | 'topP' | 'minP' | 'maxTokens' | 'depth',
    value: string
  ) => {
    if (value.trim() === '') return;

    switch (key) {
      case 'k': {
        const nextValue = Number.parseInt(value, 10);
        if (!Number.isNaN(nextValue)) updateParameters({ k: Math.max(1, nextValue) });
        break;
      }
      case 'maxTokens': {
        const nextValue = Number.parseInt(value, 10);
        if (!Number.isNaN(nextValue)) updateParameters({ maxTokens: Math.max(1, nextValue) });
        break;
      }
      case 'depth': {
        const nextValue = Number.parseInt(value, 10);
        if (!Number.isNaN(nextValue)) updateParameters({ depth: Math.max(1, nextValue) });
        break;
      }
      case 'temperature': {
        const nextValue = Number.parseFloat(value);
        if (!Number.isNaN(nextValue)) updateParameters({ temperature: Math.max(0, nextValue) });
        break;
      }
      case 'topP': {
        const nextValue = Number.parseFloat(value);
        if (!Number.isNaN(nextValue)) updateParameters({ topP: Math.min(Math.max(nextValue, 0), 1) });
        break;
      }
      case 'minP': {
        const nextValue = Number.parseFloat(value);
        if (!Number.isNaN(nextValue)) updateParameters({ minP: Math.min(Math.max(nextValue, 0), 1) });
        break;
      }
    }
  }, [updateParameters]);

  const handleLoadModel = useCallback(() => {
    if (FORCE_MOCK_CONNECTED_UI) return;
    sendMessage(TokenGenerationService.createModelLoadRequest());
  }, [sendMessage]);

  const handleGenerateTree = useCallback(() => {
    setTempSequence(null);
    setSelectedNodeId(null);
    setLoadedSourceLabel('live backend');
    setLeafLimitOverride(null);

    if (FORCE_MOCK_CONNECTED_UI) {
      setIsJsonFileLoading(true);
      fetch(MOCK_BACKEND_TREE_PATH)
        .then(response => {
          if (!response.ok) {
            throw new Error(`Failed to load mock backend tree: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          processLoadedData(data, 'live backend');
        })
        .catch(error => {
          alert(error instanceof Error ? error.message : 'Failed to load mock backend tree');
        })
        .finally(() => {
          setIsJsonFileLoading(false);
        });
      return;
    }

    sendInferenceRequest('generate_with_bfs', {
      ...parameters,
      inputText: promptText,
    });
  }, [parameters, processLoadedData, promptText, sendInferenceRequest]);

  const selectedLeafNodeId = displaySelectedSequence?.tokens[displaySelectedSequence.tokens.length - 1]?.nodeId;
  const selectedNodeEvaluation = (() => {
    if (!displayVisualTree || !selectedLeafNodeId) return null;
    const targetNode = findNodeById(displayVisualTree, selectedLeafNodeId);
    return targetNode?.evaluationCategory || targetNode?.ancestorEvaluation || null;
  })();

  const tokenStreamProps = {
    completedSequence: displaySelectedSequence,
    isNodeSelected: Boolean(selectedToken.sequenceId?.startsWith('seq-from-')),
    isGoodMarked: selectedNodeEvaluation === 'good',
    isBadMarked: selectedNodeEvaluation === 'bad',
    onMarkGood: () => {
      if (!selectedLeafNodeId) return;
      handleNodeEvaluate(selectedLeafNodeId, 'good');
    },
    onMarkBad: () => {
      if (!selectedLeafNodeId) return;
      handleNodeEvaluate(selectedLeafNodeId, 'bad');
    },
    onExpandNode: selectedNodeId
      ? () => {
          void handleRequestTokenGeneration(selectedNodeId);
        }
      : undefined,
  };

  const treeSystemSettings = useMemo(() => ({
    leftAligned: true,
    useSankeyLinks: true,
    bigTokenEnabled: true,
    useStandardTree: false,
    overviewMode: false,
    toggleVisibility: 'overview-only' as const,
    maxVisibleNodes: 7,
    centerTopNNodes: false,
  }), []);

  const effectiveConnected = FORCE_MOCK_CONNECTED_UI || webSocketState.connected;
  const effectiveError = FORCE_MOCK_CONNECTED_UI ? null : webSocketState.error;
  const effectiveModelStatus = FORCE_MOCK_CONNECTED_UI && modelStatus === 'unloaded'
    ? 'loaded'
    : modelStatus;
  const backendStatusMessage = FORCE_MOCK_CONNECTED_UI
    && loadedSourceLabel === 'live backend'
    && statusMessage === 'Tree loaded from file'
    ? 'Tree received from backend'
    : statusMessage;
  const effectiveStatusMessage = FORCE_MOCK_CONNECTED_UI
    ? (backendStatusMessage || 'Ready')
    : (webSocketState.error || statusMessage || 'Ready');
  const promptActionMessage = backendStatusMessage
    || 'Generate through the backend, or upload a saved JSON tree for inspection.';
  const effectiveMessageCount = FORCE_MOCK_CONNECTED_UI
    ? Math.max(webSocketState.messages.length, 1)
    : webSocketState.messages.length;

  const connectionTone: 'live' | 'idle' | 'warn' = effectiveConnected
    ? (effectiveModelStatus === 'error' || effectiveError ? 'warn' : 'live')
    : 'idle';
  const connectionLabel = effectiveConnected ? 'Connected' : 'Disconnected';
  const modelLabel = effectiveModelStatus === 'loading'
    ? `${effectiveModelStatus} ${loadingProgress}%`
    : effectiveModelStatus;

  return (
    <AppContainer>
      <TopBar>
        <ControlCard>
          <CardHeader>
            <CardTitle>Parameters</CardTitle>
            <ParameterHeaderActions>
              <ActionButton
                $secondary
                onClick={() => setIsAdditionalParametersOpen(prev => !prev)}
                aria-expanded={isAdditionalParametersOpen}
              >
                {isAdditionalParametersOpen ? 'Hide More' : 'More'}
              </ActionButton>
            </ParameterHeaderActions>
          </CardHeader>

          <ParameterGrid>
            <ParameterField>
              <ParameterLabelRow>
                <ParameterLabel>Temperature</ParameterLabel>
              </ParameterLabelRow>
              <ParameterInput
                type="number"
                min={0}
                step="0.1"
                value={parameters.temperature}
                onChange={e => handleParameterChange('temperature', e.target.value)}
              />
            </ParameterField>

            <ParameterField>
              <ParameterLabelRow>
                <ParameterLabel>Top P</ParameterLabel>
              </ParameterLabelRow>
              <ParameterInput
                type="number"
                min={0}
                max={1}
                step="0.01"
                value={parameters.topP}
                onChange={e => handleParameterChange('topP', e.target.value)}
              />
            </ParameterField>

            <ParameterField>
              <ParameterLabelRow>
                <ParameterLabel>Top K</ParameterLabel>
              </ParameterLabelRow>
              <ParameterInput
                type="number"
                min={1}
                value={parameters.k}
                onChange={e => handleParameterChange('k', e.target.value)}
              />
            </ParameterField>

            <ParameterField>
              <ParameterLabelRow>
                <ParameterLabel>Depth</ParameterLabel>
              </ParameterLabelRow>
              <ParameterInput
                type="number"
                min={1}
                value={parameters.depth}
                onChange={e => handleParameterChange('depth', e.target.value)}
              />
            </ParameterField>
          </ParameterGrid>

          {isAdditionalParametersOpen && (
            <AdditionalParametersRegion>
              <AdditionalParameterGrid>
                <ParameterField>
                  <ParameterLabelRow>
                    <ParameterLabel>Min P</ParameterLabel>
                  </ParameterLabelRow>
                  <ParameterInput
                    type="number"
                    min={0}
                    max={1}
                    step="0.01"
                    value={parameters.minP}
                    onChange={e => handleParameterChange('minP', e.target.value)}
                  />
                </ParameterField>

                <ParameterField>
                  <ParameterLabelRow>
                    <ParameterLabel>Max Tokens</ParameterLabel>
                  </ParameterLabelRow>
                  <ParameterInput
                    type="number"
                    min={1}
                    value={parameters.maxTokens}
                    onChange={e => handleParameterChange('maxTokens', e.target.value)}
                  />
                </ParameterField>
              </AdditionalParameterGrid>
            </AdditionalParametersRegion>
          )}
        </ControlCard>

        <BackendCard>
          <CardHeader>
            <CardTitle>Backend</CardTitle>
            <StatusPill $tone={connectionTone}>{connectionLabel}</StatusPill>
          </CardHeader>

          <BackendGrid>
            <BackendStat>
              <span>Model</span>
              <BackendStatValue>{modelLabel}</BackendStatValue>
            </BackendStat>
            <BackendStat>
              <span>Messages</span>
              <BackendStatValue>{effectiveMessageCount}</BackendStatValue>
            </BackendStat>
            <BackendStat>
              <span>Status</span>
              <BackendStatValue title={effectiveStatusMessage}>
                {effectiveStatusMessage}
              </BackendStatValue>
            </BackendStat>
          </BackendGrid>

          <ActionRow>
            <ActionButton
              type="button"
              $secondary={effectiveConnected}
              onClick={effectiveConnected ? disconnect : connect}
            >
              {effectiveConnected ? 'Disconnect' : 'Connect'}
            </ActionButton>
            <ActionButton
              type="button"
              $secondary
              disabled={!effectiveConnected || modelStatus === 'loading'}
              onClick={handleLoadModel}
            >
              Load Model
            </ActionButton>
            <ActionButton
              type="button"
              $secondary
              onClick={resetViewerState}
            >
              Clear
            </ActionButton>
          </ActionRow>
        </BackendCard>

        <PromptCard>
          <CardHeader>
            <CardTitle>Prompt</CardTitle>
            <PromptHeaderActions>
              <SourceBadge>{loadedSourceLabel}</SourceBadge>
              <FileUploadLabel htmlFor="tree-json-upload">
                Upload JSON
              </FileUploadLabel>
              <HiddenFileInput
                id="tree-json-upload"
                ref={fileInputRef}
                type="file"
                accept=".json"
                aria-hidden="true"
                tabIndex={-1}
                onChange={handleFileSelect}
              />
            </PromptHeaderActions>
          </CardHeader>

          <PromptTextArea
            value={promptText}
            onChange={e => setPromptText(e.target.value)}
            placeholder="Enter a prompt for BFS tree generation."
          />

          <ActionRow>
            <ActionButton
              type="button"
              disabled={!effectiveConnected || isGenerating}
              onClick={handleGenerateTree}
            >
              {isGenerating ? 'Generating' : 'Generate Tree'}
            </ActionButton>
            <InlineNote>
              {promptActionMessage}
            </InlineNote>
          </ActionRow>
        </PromptCard>
      </TopBar>

      <BottomArea>
        <MainContentArea>
          <ContentContainer>
            <TreeToolbar>
              <TreeToolbarMain>
                <TreeToolbarTitle>Tree</TreeToolbarTitle>
                <TreeToolbarMeta>
                  {displayVisualTree
                    ? `${displayCompletedSequences.length} responses · ${(evaluatedPathsTotal * 100).toFixed(1)}% evaluated`
                    : 'Generate a tree or upload JSON'}
                </TreeToolbarMeta>
              </TreeToolbarMain>

              <ToolbarActions>
                <ToolbarButton
                  $active={evaluationColorEnabled}
                  onClick={() => setEvaluationColorEnabled(prev => !prev)}
                >
                  Color
                </ToolbarButton>
                <ToolbarButton
                  $active={evaluationFilters.showGood}
                  $tone="good"
                  onClick={() => toggleEvaluationFilter('good')}
                >
                  Good
                </ToolbarButton>
                <ToolbarButton
                  $active={evaluationFilters.showBad}
                  $tone="bad"
                  onClick={() => toggleEvaluationFilter('bad')}
                >
                  Bad
                </ToolbarButton>
              </ToolbarActions>
            </TreeToolbar>

            <TreeVisualizerContainer>
              <TreePanelControls>
                <TreeSettingsAnchor>
                  {isTreeSettingsOpen && (
                    <TreeSettingsPanel>
                      <SliderSetting
                        label="Overview depth"
                        value={overviewDepth}
                        min={MIN_OVERVIEW_DEPTH}
                        max={MAX_OVERVIEW_DEPTH}
                        step={1}
                        onChange={setOverviewDepth}
                        formatValue={value => `D${value}`}
                      />
                      <SliderSetting
                        label="Filtering min prob"
                        value={filteringMinProb}
                        min={MIN_FILTERING_MIN_PROB}
                        max={MAX_FILTERING_MIN_PROB}
                        step={0.01}
                        onChange={setFilteringMinProb}
                        formatValue={formatPercent}
                      />
                      <SliderSetting
                        label="Filtering max branches"
                        value={filteringMaxBranches}
                        min={MIN_FILTERING_MAX_BRANCHES}
                        max={MAX_FILTERING_MAX_BRANCHES}
                        step={1}
                        onChange={setFilteringMaxBranches}
                      />
                    </TreeSettingsPanel>
                  )}
                  <IconToolbarButton
                    type="button"
                    $active={isOverviewEnabled}
                    onClick={() => setIsOverviewEnabled(prev => !prev)}
                    aria-label={isOverviewEnabled ? 'Turn overview off' : 'Turn overview on'}
                    title={isOverviewEnabled ? 'Turn overview off' : 'Turn overview on'}
                  >
                    <TreeModeIcon />
                  </IconToolbarButton>
                  <IconToolbarButton
                    type="button"
                    $active={isTreeSettingsOpen}
                    onClick={() => setIsTreeSettingsOpen(prev => !prev)}
                    aria-label={isTreeSettingsOpen ? 'Close tree settings' : 'Open tree settings'}
                    title="Tree settings"
                  >
                    <GearIcon />
                  </IconToolbarButton>
                </TreeSettingsAnchor>
              </TreePanelControls>

              {displayVisualTree ? (
                <TokenTreeVisualizer
                  node={displayVisualTree}
                  selectedToken={selectedToken}
                  completedSequences={displayCompletedSequences}
                  tempSequence={tempSequence}
                  onNodePin={handleNodePin}
                  onNodeEvaluate={handleNodeEvaluate}
                  onGenerateFromNode={handleRequestTokenGeneration}
                  isNodeGenerating={isNodeGenerating}
                  onTokenSelect={handleTreeNodeSelect}
                  evaluatedNodeIds={evaluatedNodeIds}
                  evaluatedPathsTotal={evaluatedPathsTotal}
                  evaluationFilters={evaluationFilters}
                  evaluationColorEnabled={evaluationColorEnabled}
                  systemSettings={treeSystemSettings}
                  externalOverviewEnabled={isOverviewEnabled}
                  externalOverviewMaxDepth={overviewDepth}
                  filteringMinProb={filteringMinProb}
                  filteringMaxBranches={filteringMaxBranches}
                  showLayoutToggles={false}
                />
              ) : (
                <EmptyState>
                  No tree data loaded yet.
                </EmptyState>
              )}
            </TreeVisualizerContainer>
          </ContentContainer>
        </MainContentArea>

        <StreamCard>
          <TokenStreamPanel key={panelResetKey} {...tokenStreamProps} />
        </StreamCard>
      </BottomArea>

      {isJsonFileLoading && (
        <GlobalLoadingOverlay>
          <GlobalLoadingCard role="status" aria-live="polite">
            <LoadingSpinner />
            JSON loading in progress...
          </GlobalLoadingCard>
        </GlobalLoadingOverlay>
      )}
    </AppContainer>
  );
}

export default App;
