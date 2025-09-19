export type InferenceMode = 'explore_node' | 'load_model' | 'generate_with_smc';

export interface InferenceRequest {
  type: InferenceMode;
  input_text: string;
  k: number;
  temperature: number;
  top_p: number;
  min_p: number;
  max_tokens?: number;
  depth?: number;
  node_id?: string; 
  request_id?: string;
}

export interface ApiNode {
  id: string | number; //
  token_id?: number; 
  text: string;
  prob: number;
  score: number;
  depth?: number;
  children: ApiNode[];
}

export interface TokenTreeResponse {
  request_id: string;
  type: 'tree_result';
  tree: {
    id: string | number;
    token_id?: number; 
    text: string;
    prob: number;
    score: number;
    depth?: number;
    children: ApiNode[];
  };
}

export interface ModelStatusResponse {
  request_id: string;
  type: 'model_status';
  status: 'loading' | 'loaded' | 'unloaded' | 'error';
  message?: string;
}

export interface LoadingStatusResponse {
  request_id: string;
  type: 'loading_status';
  status: 'loading';
  progress: number;
  message?: string;
}

export interface GenerationStatusResponse {
  request_id: string;
  type: 'generation_status';
  status: 'started' | 'completed';
  message?: string;
}

export interface ErrorResponse {
  request_id?: string;
  type: 'error';
  message: string;
}

export interface UpdateResponse {
  type: 'update';
  request_id?: string;
  tree: {
    id: string | number;
    token_id?: number;
    text: string;
    prob: number;
    score: number;
    depth?: number;
    children: ApiNode[];
  };
}

export type WebSocketResponse = TokenTreeResponse | ModelStatusResponse | LoadingStatusResponse | GenerationStatusResponse | ErrorResponse | UpdateResponse;

export interface WebSocketState {
  connected: boolean;
  error: string | null;
  messages: Array<{
    direction: 'sent' | 'received';
    timestamp: number;
    data: any;
  }>;
}

export interface VisualNode {
  id: string;
  token_id?: number;
  path: string;
  token: string;
  prob: number;
  cumulativeProb?: number;
  score?: number;
  depth?: number; 
  children: VisualNode[];
  isFolded?: boolean;  
  isUserFolded?: boolean;
  // for big tokens
  isExpanded?: boolean; 
  mergedNodes?: {
    tokens: string[];
    probs: number[];
    nodeIds?: string[];
  };
  nodeState?: 'generating' | 'completed';
  lastGenerationRequestTime?: number;
  isPinned?: boolean;
  isEvaluated?: boolean;
  evaluationCategory?: 'good' | 'bad' | null; 
  ancestorEvaluation?: 'good' | 'bad' | null; 
  _foldStateBackup?: Record<string, {
    isFolded: boolean | undefined,
    isUserFolded: boolean | undefined
  }>;
  foldedSiblingCount?: number; 
  isFiltered?: boolean; 
}

export interface CompletedSequence {
  id: string;
  text: string;
  tokens: TokenInfo[];
  totalProb: number;
  path: string;
}

export interface TokenInfo {
  token: string;
  nodeId: string;
  prob: number;
  startIndex: number;
  endIndex: number;
}

export interface TokenSelection {
  sequenceId: string | null;
  tokenIndex: number | null;
}

export interface EvaluatedNodes {
  good: Set<string>;
  bad: Set<string>;
}