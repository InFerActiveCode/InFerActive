import { useState, useEffect, useCallback, useRef } from 'react';
import {
  VisualNode,
  TokenSelection,
  CompletedSequence,
  TokenTreeResponse,
  WebSocketState,
  InferenceMode,
  WebSocketResponse,
  EvaluatedNodes
} from '../types/types';
import {
  transformToVisualTree,
  findAlternativeTokens,
  findNodeById,
  mergeTrees,
  findParentNode
} from '../utils/treeTransform';
import { TokenGenerationService } from '../services/tokenGenService';
import { calculateEvaluatedPathsProbability } from '../components/tokenTreeUtils';

interface UseTreeManagerProps {
  webSocketState: WebSocketState;
  sendMessage: (message: any) => void;
  clearMessages: () => void;
  tempSequence?: CompletedSequence | null;
}


export interface InferenceParameters {
  inputText: string;
  k: number;
  temperature: number;
  topP: number;
  minP: number;
  maxTokens: number;
  depth: number;
}

interface TreeManagerState {
  visualTree: VisualNode | null;
  completedSequences: CompletedSequence[];
  selectedToken: TokenSelection;
  alternativeTokens: Array<{ token: string; probability: number; nodeId: string }>;
  parameters: InferenceParameters;
  currentRequestId: string | null;
  modelStatus: 'unloaded' | 'loading' | 'loaded' | 'error';
  loadingProgress: number;
  statusMessage: string;
  isGenerating: boolean;
  evaluatedNodes: EvaluatedNodes;
  evaluatedPathsTotal: number;
  evaluationFilters: {
    showGood: boolean;
    showBad: boolean;
    showUnmarked: boolean;
  };
}

export const useTreeManager = ({
  webSocketState,
  sendMessage,
  clearMessages,
  tempSequence
}: UseTreeManagerProps) => {
  
  const GENERATION_TIMEOUT = 30000; 

  
  const [state, setState] = useState<TreeManagerState>({
    visualTree: null,
    completedSequences: [],
    selectedToken: { sequenceId: null, tokenIndex: null },
    alternativeTokens: [],
    parameters: TokenGenerationService.getDefaultParameters(), 
    currentRequestId: null,
    modelStatus: 'unloaded', 
    loadingProgress: 0,
    statusMessage: '',
    isGenerating: false,
    evaluatedNodes: { good: new Set<string>(), bad: new Set<string>() },
    evaluatedPathsTotal: 0,
    evaluationFilters: {
      showGood: true,
      showBad: true,
      showUnmarked: true
    },
  });

  
  const [selectionUpdateInfo, setSelectionUpdateInfo] = useState<{
    newSelection: TokenSelection
  } | null>(null);

  
  const processedMessageCountRef = useRef(0);

  
  const tokenGenerationRequestRef = useRef<(
    nodeId: string,
    requestId?: string
  ) => Promise<boolean>>(async () => Promise.resolve(false));

  
  useEffect(() => {
    if (webSocketState.connected) {
      
      const checkRequest = TokenGenerationService.createModelStatusCheckRequest(state.currentRequestId || undefined);
      sendMessage(checkRequest);
    }
  }, [webSocketState.connected, sendMessage, state.currentRequestId]);

  
  useEffect(() => {
    
    const allEvaluatedNodeIds = new Set([
      ...Array.from(state.evaluatedNodes.good),
      ...Array.from(state.evaluatedNodes.bad)
    ]);
    
    const totalProbability = calculateEvaluatedPathsProbability(
      state.visualTree,
      allEvaluatedNodeIds
    );
    
    setState(prevState => ({
      ...prevState,
      evaluatedPathsTotal: totalProbability
    }));
  }, [state.visualTree, state.evaluatedNodes]);

  const updateParameters = useCallback((newParams: Partial<InferenceParameters>) => {
    setState(prevState => ({
      ...prevState,
      parameters: {
        ...prevState.parameters,
        ...newParams
      }
    }));
  }, []);

  useEffect(() => {
    const handleTreeFileLoaded = () => {
      if ((window as any).__loadedTreeData) {
        const loadedTree = (window as any).__loadedTreeData;
        delete (window as any).__loadedTreeData;
        
        
        setState(prevState => {
          
          const transformResult = transformToVisualTree(
            loadedTree.tree,
            null,  
            1.0    
          );
          
          return {
            ...prevState,
            visualTree: transformResult.tree,
            completedSequences: transformResult.completedSequences,
            currentRequestId: loadedTree.request_id,
            isGenerating: false,
            modelStatus: 'loaded',
            statusMessage: 'Tree loaded from file'
          };
        });
      }
    };
    
    window.addEventListener('treeFileLoaded', handleTreeFileLoaded);
    return () => {
      window.removeEventListener('treeFileLoaded', handleTreeFileLoaded);
    };
  }, []);

  useEffect(() => {
    
    if (
      webSocketState.messages.length === 0 ||
      webSocketState.messages.length <= processedMessageCountRef.current
    ) return;

    const lastMessage = webSocketState.messages[webSocketState.messages.length - 1];
    if (lastMessage.direction !== 'received') {
      processedMessageCountRef.current = webSocketState.messages.length;
      return;
    }

    
    try {
      const response = lastMessage.data as WebSocketResponse;

      
      switch (response.type) {
        case 'model_status':
          setState(prevState => ({
            ...prevState,
            modelStatus: response.status,
            statusMessage: response.message || ''
          }));
          break;

        case 'loading_status':
          setState(prevState => ({
            ...prevState,
            modelStatus: 'loading',
            loadingProgress: response.progress,
            statusMessage: response.message || ''
          }));
          break;

        case 'generation_status':
          setState(prevState => ({
            ...prevState,
            isGenerating: response.status === 'started',
            statusMessage: response.message || ''
          }));
          break;

        case 'error':
          setState(prevState => ({
            ...prevState,
            modelStatus: 'error',
            statusMessage: response.message,
            isGenerating: false
          }));
          break;

        case 'tree_result':
        case 'update':  
          
          const currentRequestId = !state.currentRequestId && (response as any).request_id ?
            (response as any).request_id : state.currentRequestId;

          
          setState(prevState => {
        
        
        let parentCumulativeProb = 1.0;
        if (prevState.visualTree && (response as any).tree.id !== 'root') {
          const parentNode = findParentNode(prevState.visualTree, (response as any).tree.id);
          if (parentNode) {
            parentCumulativeProb = parentNode.cumulativeProb || 1.0;
          }
        }
        
        const transformResult = transformToVisualTree(
          (response as any).tree, 
          prevState.visualTree,
          parentCumulativeProb
        );

        const newSequences = transformResult.completedSequences;

        
        const newTree = mergeTrees(
          prevState.visualTree,
          transformResult.tree,
          transformResult.isSubtree
        );

        
        if (newTree) {
          const updateNodeStates = (node: VisualNode) => {
            for (const child of node.children) {
              const nodeInResponse = findNodeById(transformResult.tree, child.id);

              if (nodeInResponse && child.nodeState === 'generating') {
                child.nodeState = 'completed';
              }

              updateNodeStates(child);
            }
          };

          updateNodeStates(newTree);
        }

        
        const existingIds = new Set(prevState.completedSequences.map(seq => seq.id));
        const newSequencesToAdd = newSequences.filter(seq => !existingIds.has(seq.id));

        const updatedSequences = [...prevState.completedSequences, ...newSequencesToAdd]
          .sort((a, b) => b.totalProb - a.totalProb);

        
        if (newSequencesToAdd.length > 0 && newTree) {
          
          const findGeneratingNodeIds = (node: VisualNode): string[] => {
            const result: string[] = [];
            if (node.nodeState === 'generating') {
              result.push(node.id);
            }
            for (const child of node.children) {
              result.push(...findGeneratingNodeIds(child));
            }
            return result;
          };

          const generatingNodeIds = findGeneratingNodeIds(newTree);

          
          if (generatingNodeIds.length > 0) {
            for (const nodeId of generatingNodeIds) {
              const newSequenceWithGeneratingToken = newSequences.find(seq =>
                seq.tokens.some(token => token.nodeId === nodeId)
              );

              if (newSequenceWithGeneratingToken) {
                const tokenIndex = newSequenceWithGeneratingToken.tokens.findIndex(
                  token => token.nodeId === nodeId
                );

                if (tokenIndex >= 0) {
                  
                  setSelectionUpdateInfo({
                    newSelection: {
                      sequenceId: newSequenceWithGeneratingToken.id,
                      tokenIndex: tokenIndex
                    }
                  });
                  break;
                }
              }
            }
          }
        }

        
        const newAlternativeTokens = calculateAlternativeTokens(
          newTree,
          updatedSequences,
          prevState.selectedToken,
          tempSequence
        );

            
            return {
              ...prevState,
              visualTree: newTree,
              completedSequences: updatedSequences,
              alternativeTokens: newAlternativeTokens,
              currentRequestId,
              isGenerating: false
            };
          });
          break;

        default:
          console.warn('Unknown message type:', (response as any).type);
      }

      
      processedMessageCountRef.current = webSocketState.messages.length;

    } catch (error) {
      console.error('error:', error);
      processedMessageCountRef.current = webSocketState.messages.length;
    }
  }, [webSocketState.messages, state.currentRequestId, tempSequence]);

  useEffect(() => {
    if (selectionUpdateInfo) {
      setState(prevState => ({
        ...prevState,
        selectedToken: selectionUpdateInfo.newSelection,
        alternativeTokens: calculateAlternativeTokens(
          prevState.visualTree,
          prevState.completedSequences,
          selectionUpdateInfo.newSelection,
          tempSequence
        )
      }));

      setSelectionUpdateInfo(null);
    }
  }, [selectionUpdateInfo, tempSequence]);

  const calculateAlternativeTokens = (
    tree: VisualNode | null,
    sequences: CompletedSequence[],
    selection: TokenSelection,
    tempSeq?: CompletedSequence | null
  ) => {
    if (selection.sequenceId && selection.tokenIndex !== null && tree) {
      return findAlternativeTokens(
        tree,
        sequences,
        selection.sequenceId,
        selection.tokenIndex,
        tempSeq
      );
    }
    return [];
  };

  const resetAll = useCallback(() => {
    clearMessages();
    setState(prevState => ({
      ...prevState,
      visualTree: null,
      completedSequences: [],
      selectedToken: { sequenceId: null, tokenIndex: null },
      alternativeTokens: [],
      currentRequestId: null
    }));
    setSelectionUpdateInfo(null);
    processedMessageCountRef.current = 0;
  }, [clearMessages]);

  const handleTokenSelect = useCallback((sequenceId: string, tokenIndex: number) => {
    setState(prevState => ({
      ...prevState,
      selectedToken: { sequenceId, tokenIndex },
      alternativeTokens: calculateAlternativeTokens(
        prevState.visualTree,
        prevState.completedSequences,
        { sequenceId, tokenIndex },
        tempSequence
      )
    }));
  }, [tempSequence]);

  const handleSequenceSelect = useCallback((sequence: CompletedSequence) => {
    
    const lastValidIndex = sequence.tokens.reduce((lastIdx, token, idx) => {
      if (!['<|eot_id|>', '</s>', '<|endoftext|>'].includes(token.token)) {
        return idx;
      }
      return lastIdx;
    }, 0);

    setState(prevState => {
      const newSelection = {
        sequenceId: sequence.id,
        tokenIndex: lastValidIndex
      };

      return {
        ...prevState,
        selectedToken: newSelection,
        alternativeTokens: calculateAlternativeTokens(
          prevState.visualTree,
          prevState.completedSequences,
          newSelection,
          tempSequence
        )
      };
    });
  }, [tempSequence]);

  const getSelectedSequence = useCallback((): CompletedSequence | null => {
    return state.completedSequences.find(
      seq => seq.id === state.selectedToken.sequenceId
    ) || null;
  }, [state.selectedToken.sequenceId, state.completedSequences]);

  const isNodeGenerating = useCallback((nodeId: string): boolean => {
    if (!state.visualTree) return false;
    const node = findNodeById(state.visualTree, nodeId);
    return TokenGenerationService.isNodeGenerating(node);
  }, [state.visualTree]);

  const isNodeCompleted = useCallback((nodeId: string): boolean => {
    if (!state.visualTree) return false;
    const node = findNodeById(state.visualTree, nodeId);
    return TokenGenerationService.isNodeCompleted(node);
  }, [state.visualTree]);


  const canGenerateFromNode = useCallback((nodeId: string): boolean => {
    if (!state.visualTree) return false;
    const node = findNodeById(state.visualTree, nodeId);
    return TokenGenerationService.canGenerateFromNode(node, GENERATION_TIMEOUT);
  }, [state.visualTree, GENERATION_TIMEOUT]);


const handleNodePin = useCallback((nodeId: string) => {
  setState(prevState => {
    if (!prevState.visualTree) return prevState;
    
    const updatedTree = {...prevState.visualTree};
    const targetNode = findNodeById(updatedTree, nodeId);
    
    if (targetNode) {
      
      const newPinState = !targetNode.isPinned;
      
      
      if (newPinState) {
        
        targetNode.isPinned = true;
        
        
        const pathToRoot: string[] = [];
        let currentNode = targetNode;
        let parentNode = findParentNode(updatedTree, currentNode.id);
        
        
        pathToRoot.push(currentNode.id);
        while (parentNode) {
          currentNode = parentNode;
          pathToRoot.push(currentNode.id);
          parentNode = findParentNode(updatedTree, currentNode.id);
        }
        
        
        pathToRoot.reverse();
        
        
        const foldStateBackup: Record<string, {
          isFolded: boolean | undefined,
          isUserFolded: boolean | undefined
        }> = {};
        
        
        for (let i = 0; i < pathToRoot.length - 1; i++) {
          const parentId = pathToRoot[i];
          const childId = pathToRoot[i + 1];
          
          const parent = findNodeById(updatedTree, parentId);
          
          if (parent && parent.children) {
            
            parent.children.forEach(child => {
              if (child.id !== childId) {
                
                foldStateBackup[child.id] = {
                  isFolded: child.isFolded,
                  isUserFolded: child.isUserFolded
                };
                
                child.isFolded = true;
                child.isUserFolded = true;
              }
            });
          }
        }
        
        
        targetNode._foldStateBackup = foldStateBackup;
      } 
      
      else {
        
        targetNode.isPinned = false;
        
        
        if (targetNode._foldStateBackup) {
          
          
          Object.keys(targetNode._foldStateBackup).forEach(nodeId => {
            const state = targetNode._foldStateBackup![nodeId];
            const node = findNodeById(updatedTree, nodeId);
            if (node) {
              node.isFolded = state.isFolded;
              node.isUserFolded = state.isUserFolded;
            }
          });
          
          
          delete targetNode._foldStateBackup;
        }
      }
    }
    
    return {
      ...prevState,
      visualTree: updatedTree
    };
  });
}, []);

  const markDescendantsAsRestricted = (node: VisualNode, parentCategory: 'good' | 'bad' | null, evaluatedNodeIds?: Set<string>) => {
    if (node.children) {
      node.children.forEach(child => {
        if (parentCategory && child.isEvaluated && evaluatedNodeIds) {
          
          child.isEvaluated = false;
          evaluatedNodeIds.delete(child.id);
        }
        
        child.ancestorEvaluation = parentCategory;
        markDescendantsAsRestricted(child, parentCategory, evaluatedNodeIds);
      });
    }
  };

  const markDescendantsAsRestrictedWithCategories = (node: VisualNode, parentCategory: 'good' | 'bad' | null, evaluatedNodes?: EvaluatedNodes) => {
    if (node.children) {
      node.children.forEach(child => {
        if (parentCategory && child.evaluationCategory && evaluatedNodes) {
          
          child.isEvaluated = false;
          evaluatedNodes[child.evaluationCategory].delete(child.id);
          child.evaluationCategory = null;
        }
        
        child.ancestorEvaluation = parentCategory;
        markDescendantsAsRestrictedWithCategories(child, parentCategory, evaluatedNodes);
      });
    }
  };


  const checkAndConsolidateSiblings = (
    tree: VisualNode,
    node: VisualNode,
    category: 'good' | 'bad',
    evaluatedNodes: EvaluatedNodes
  ): string | null => {
    const parent = findParentNode(tree, node.id);
    if (!parent || parent.id === 'root') return null;
    
    const allSameCategory = parent.children.every(
      child => child.evaluationCategory === category
    );
    
    if (allSameCategory) {
      
      parent.children.forEach(child => {
        child.evaluationCategory = null;
        child.isEvaluated = false;
        evaluatedNodes[category].delete(child.id);
        
      });
      
      
      const parentHasSinglePath = findAndMarkTopSinglePathAncestor(tree, parent, category, evaluatedNodes);
      
      if (!parentHasSinglePath) {
        
        parent.evaluationCategory = category;
        parent.isEvaluated = true;
        evaluatedNodes[category].add(parent.id);
        markDescendantsAsRestrictedWithCategories(parent, category, evaluatedNodes);
        
        
        const higherParentId = checkAndConsolidateSiblings(tree, parent, category, evaluatedNodes);
        
        return higherParentId || parent.id;
      }
      
      return parentHasSinglePath;
    }
    return null;
  };

  const findAndMarkTopSinglePathAncestor = (
    tree: VisualNode, 
    node: VisualNode,
    category: 'good' | 'bad',
    evaluatedNodes: EvaluatedNodes
  ): string | null => {  
    
    const parentNode = findParentNode(tree, node.id);
    if (!parentNode || parentNode.id === 'root') return null;
    
    
    if (parentNode.children.length !== 1) return null;
    
    
    let currentNodeId = node.id;
    let topSingleParentId = null;
    
    while (true) {
      const parent = findParentNode(tree, currentNodeId);
      
      if (!parent || parent.id === 'root') break;
      
      
      if (parent.children.length === 1) {
        topSingleParentId = parent.id;
        currentNodeId = parent.id;
      } else {
        
        break;
      }
    }
    
    
    if (topSingleParentId) {
      const topParent = findNodeById(tree, topSingleParentId);
      if (topParent) {
        
        evaluatedNodes['good'].delete(topSingleParentId);
        evaluatedNodes['bad'].delete(topSingleParentId);
        topParent.evaluationCategory = category;
        topParent.isEvaluated = true;
        evaluatedNodes[category].add(topSingleParentId);
        
        
        markDescendantsAsRestrictedWithCategories(topParent, category, evaluatedNodes);
        
        
        const consolidatedParentId = checkAndConsolidateSiblings(tree, topParent, category, evaluatedNodes);
        
        return consolidatedParentId || topSingleParentId;
      }
      return topSingleParentId; 
    }
    
    return null; 
  };

  const findMarkedAncestorInSinglePath = (
    tree: VisualNode, 
    node: VisualNode,
    category: 'good' | 'bad'
  ): string | null => {
    let currentNodeId = node.id;
    
    while (true) {
      const parent = findParentNode(tree, currentNodeId);
      
      if (!parent || parent.id === 'root') break;
      
      
      if (parent.evaluationCategory === category && parent.children.length === 1) {
        return parent.id;
      }
      
      
      if (parent.children.length !== 1) break;
      
      currentNodeId = parent.id;
    }
    
    return null;
  };

  
  const handleNodeEvaluate = useCallback((nodeId: string, category: 'good' | 'bad') => {
    setState(prevState => {
      if (!prevState.visualTree) return prevState;
      
      const updatedTree = { ...prevState.visualTree };
      const targetNode = findNodeById(updatedTree, nodeId);
      
      if (!targetNode) return prevState;
      
      const newEvaluatedNodes = {
        good: new Set(prevState.evaluatedNodes.good),
        bad: new Set(prevState.evaluatedNodes.bad)
      };
      
      let nodeToAddToRecent: string | null = null;
      
      
      if (targetNode.evaluationCategory === category) {
        targetNode.evaluationCategory = null;
        targetNode.isEvaluated = false;
        newEvaluatedNodes[category].delete(nodeId);
        
        markDescendantsAsRestricted(targetNode, null);
      } else if (targetNode.ancestorEvaluation === category) {
        
        const markedAncestorId = findMarkedAncestorInSinglePath(updatedTree, targetNode, category);
        if (markedAncestorId) {
          const markedAncestor = findNodeById(updatedTree, markedAncestorId);
          if (markedAncestor) {
            markedAncestor.evaluationCategory = null;
            markedAncestor.isEvaluated = false;
            newEvaluatedNodes[category].delete(markedAncestorId);
            
            markDescendantsAsRestricted(markedAncestor, null);
          }
        }
      } else {
        
        if (targetNode.evaluationCategory) {
          newEvaluatedNodes[targetNode.evaluationCategory].delete(nodeId);
        }
        
        const topSingleParentId = findAndMarkTopSinglePathAncestor(updatedTree, targetNode, category, newEvaluatedNodes);
        
        if (!topSingleParentId) {
          
          targetNode.evaluationCategory = category;
          targetNode.isEvaluated = true;
          newEvaluatedNodes[category].add(nodeId);
          
          
          markDescendantsAsRestrictedWithCategories(targetNode, category, newEvaluatedNodes);
          
          
          const consolidatedParentId = checkAndConsolidateSiblings(updatedTree, targetNode, category, newEvaluatedNodes);
          
          
          nodeToAddToRecent = consolidatedParentId || nodeId;
        } else {
          
          nodeToAddToRecent = topSingleParentId;
        }
      }
      
      
      return {
        ...prevState,
        visualTree: updatedTree,
        evaluatedNodes: newEvaluatedNodes,
      };
    });
  }, []);

  const toggleEvaluationFilter = useCallback((filter: 'good' | 'bad' | 'unmarked') => {
    setState(prevState => {
      const key = filter === 'good' ? 'showGood' : 
                  filter === 'bad' ? 'showBad' : 
                  'showUnmarked';
      
      return {
        ...prevState,
        evaluationFilters: {
          ...prevState.evaluationFilters,
          [key]: !prevState.evaluationFilters[key]
        }
      };
    });
  }, []);


  const handleRequestTokenGeneration = useCallback(async (
    nodeId: string,
    customParams?: Partial<InferenceParameters & { requestId?: string }>
  ): Promise<CompletedSequence | null> => {

    if (!state.visualTree) {
      console.warn('no tree');
      return null;
    }

    
    if (!canGenerateFromNode(nodeId)) {
      console.warn(`can't generate from node ${nodeId}`);;

      
      if (isNodeCompleted(nodeId)) {
        const sequence = state.completedSequences.find(seq =>
          seq.tokens.some(token => token.nodeId === nodeId)
        );
        if (sequence) return sequence;
      }

      return null;
    }

    
    setState(prevState => {
      if (!prevState.visualTree) return prevState;

      const updatedTree = { ...prevState.visualTree };
      const targetNode = findNodeById(updatedTree, nodeId);

      if (targetNode) {
        targetNode.nodeState = 'generating';
        targetNode.lastGenerationRequestTime = Date.now();
      }

      return {
        ...prevState,
        visualTree: updatedTree
      };
    });

    
    try {
      
      const mergedParams = {
        ...state.parameters,
        ...customParams
      };

      
      const request = TokenGenerationService.createNodeExplorationRequest(
        mergedParams.inputText,
        nodeId,
        {
          k: mergedParams.k,
          temperature: mergedParams.temperature,
          topP: mergedParams.topP,
          minP: mergedParams.minP,
          maxTokens: mergedParams.maxTokens,
          requestId: mergedParams.requestId || (state.currentRequestId || undefined)
        }
      );

      
      sendMessage(request);

      
      
      
      return null;

    } catch (error) {
      console.error('error:', error);

      
      setState(prevState => {
        if (!prevState.visualTree) return prevState;

        const updatedTree = { ...prevState.visualTree };
        const targetNode = findNodeById(updatedTree, nodeId);

        if (targetNode) {
          targetNode.nodeState = undefined;
          targetNode.lastGenerationRequestTime = undefined;
        }

        return {
          ...prevState,
          visualTree: updatedTree
        };
      });

      return null;
    }
  }, [
    state.visualTree,
    state.completedSequences,
    state.parameters,
    state.currentRequestId,
    canGenerateFromNode,
    isNodeCompleted,
    sendMessage
  ]);


  const sendInferenceRequest = useCallback((
    mode: InferenceMode,
    customParams?: Partial<InferenceParameters>
  ) => {
    
    
    if (state.modelStatus === 'unloaded') {
      console.log('Model status is unloaded, sending load request');
      const loadRequest = TokenGenerationService.createModelLoadRequest(state.currentRequestId || undefined);
      sendMessage(loadRequest);
      
      return;
    }

    
    const mergedParams = {
      ...state.parameters,
      ...customParams
    };

    
    const request = TokenGenerationService.createInferenceRequest(
      mode,
      mergedParams.inputText,
      {
        k: mergedParams.k,
        temperature: mergedParams.temperature,
        topP: mergedParams.topP,
        minP: mergedParams.minP,
        maxTokens: mergedParams.maxTokens,
        depth: mergedParams.depth,
        requestId: state.currentRequestId || undefined
      }
    );

    
    setState(prevState => ({
      ...prevState,
      parameters: mergedParams
    }));

    
    sendMessage(request);

    return request;
  }, [state.parameters, state.currentRequestId, state.modelStatus, sendMessage]);

  useEffect(() => {
    tokenGenerationRequestRef.current = async (nodeId: string, requestId?: string) => {
      try {
        await handleRequestTokenGeneration(nodeId, { requestId });
        return true;
      } catch (error) {
        console.error('error:', error);
        return false;
      }
    };
  }, [handleRequestTokenGeneration]);

  return {
    
    visualTree: state.visualTree,
    completedSequences: state.completedSequences,
    selectedToken: state.selectedToken,
    alternativeTokens: state.alternativeTokens,
    parameters: state.parameters,
    modelStatus: state.modelStatus,
    loadingProgress: state.loadingProgress,
    statusMessage: state.statusMessage,
    isGenerating: state.isGenerating,
    evaluatedNodeIds: state.evaluatedNodes,
    evaluatedPathsTotal: state.evaluatedPathsTotal,
    evaluationFilters: state.evaluationFilters,

    
    resetAll,
    handleTokenSelect,
    handleSequenceSelect,
    handleNodePin,
    handleNodeEvaluate,
    toggleEvaluationFilter,
    handleRequestTokenGeneration,
    sendInferenceRequest,
    updateParameters,

    
    getSelectedSequence,
    canGenerateFromNode,
    isNodeGenerating,
    isNodeCompleted,

    
    tokenGenerationRequestRef
  };
};