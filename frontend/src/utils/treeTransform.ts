import { VisualNode, CompletedSequence, TokenInfo, TokenTreeResponse, ApiNode } from '../types/types';

export function transformToVisualTree(
  apiTree: any, 
  parentTree: VisualNode | null = null, 
  parentCumulativeProb: number = 1.0 
): {
  tree: VisualNode;
  completedSequences: CompletedSequence[];
  isSubtree: boolean; 
} {
  
  const isSubtree = apiTree.id && apiTree.id !== 'root';
  
  const root: VisualNode = {
    id: `${apiTree.id || 'root'}`, 
    token_id: apiTree.token_id,
    path: '',
    token: apiTree.id === 'root' ? 'root' : (apiTree.text || ''), 
    prob: apiTree.prob || 1,
    cumulativeProb: parentCumulativeProb * (apiTree.prob || 1), 
    score: apiTree.score || 1,
    depth: apiTree.depth || 0,
    children: []
  };

  
  let parentNode: VisualNode | null = null;
  if (parentTree && isSubtree) {
    const nodeId = root.id
    parentNode = findNodeById(parentTree, `${nodeId}`);
    
    
    if (!parentNode) {
      parentNode = findParentNode(parentTree, `${nodeId}`);
    }
  }

  const completedSequences: CompletedSequence[] = [];
  const nodeMap = new Map<string, VisualNode>();
  nodeMap.set(root.id, root);

  
  function processNode(apiNode: any, parentNode: VisualNode, parentPath: string, parentHasEvaluatedAncestor: boolean = false) {
    
    if (Array.isArray(apiNode.children)) {
      for (const childNode of apiNode.children) {
        const tokenText = childNode.text || '';
        const nodePath = parentPath + tokenText;
        
        const nodeId = `${childNode.id}`;
        
        
        const existingParent = parentTree ? findNodeById(parentTree, parentNode.id) : null;
        
        
        const ancestorEvaluation = parentHasEvaluatedAncestor ? 
                                  (existingParent?.evaluationCategory || existingParent?.ancestorEvaluation || null) :
                                  (existingParent?.ancestorEvaluation || null);
        
        const visualNode: VisualNode = {
          id: nodeId,
          token_id: childNode.token_id,
          path: nodePath,
          token: tokenText,
          prob: childNode.prob,
          cumulativeProb: (parentNode.cumulativeProb || 1) * childNode.prob, 
          score: childNode.score,
          depth: childNode.depth || 0,
          children: [],
          
          ancestorEvaluation
        };
        
        nodeMap.set(nodeId, visualNode);
        parentNode.children.push(visualNode);
        
        
        processNode(childNode, visualNode, nodePath, ancestorEvaluation !== null);
        
        
        if (tokenText === '<|eot_id|>' || tokenText === '</s>' || tokenText === '<|endoftext|>') {
          createCompletedSequence(visualNode, nodePath, parentTree);
        }
      }
    }
  }

  
  function createCompletedSequence(node: VisualNode, path: string, parentTree: VisualNode | null = null) {
    const sequence: TokenInfo[] = [];
    let totalProb = 1;
    let text = '';
    
    
    const nodeChain: VisualNode[] = [];
    let currentNode: VisualNode | null = node;
    
    
    while (currentNode) {
      nodeChain.unshift(currentNode);
      if (currentNode.id === root.id) break;
      
      currentNode = findParentNode(root, currentNode.id);
      if (!currentNode) break;
    }
    
    
    if (isSubtree && parentTree && nodeChain.length > 0) {
      const headNode = nodeChain[0]; 
      
      
      let parentHeadNode = findParentNode(parentTree, headNode.id);
      
      
      const parentChain: VisualNode[] = [];
      while (parentHeadNode) {
        parentChain.unshift(parentHeadNode);
        if (parentHeadNode.id === 'root') break;
        
        parentHeadNode = findParentNode(parentTree, parentHeadNode.id);
        if (!parentHeadNode) break;
      }
      
      
      nodeChain.unshift(...parentChain);
    }
    
    
    for (const chainNode of nodeChain) {
      if (chainNode.id === 'root' && chainNode.token === 'root') continue;
      
      const startIndex = text.length;
      text += chainNode.token;
      
      sequence.push({
        token: chainNode.token,
        nodeId: chainNode.id,
        prob: chainNode.prob,
        startIndex,
        endIndex: startIndex + chainNode.token.length
      });
      
      totalProb *= chainNode.prob;
    }

    completedSequences.push({
      id: `seq-${node.id}`, 
      text,
      tokens: sequence,
      totalProb,
      path
    });
  }

  
  const existingRoot = parentTree ? findNodeById(parentTree, root.id) : null;
  const rootHasEvaluatedAncestor = existingRoot?.isEvaluated || (existingRoot?.ancestorEvaluation !== null && existingRoot?.ancestorEvaluation !== undefined) || false;
  
  
  processNode(apiTree, root, '', rootHasEvaluatedAncestor);

  
  completedSequences.sort((a, b) => b.totalProb - a.totalProb);

  return {
    tree: root,
    completedSequences,
    isSubtree
  };
}


export function findNodeById(root: VisualNode, nodeId: string): VisualNode | null {
  if (root.id === nodeId) return root;
  
  for (const child of root.children) {
    const found = findNodeById(child, nodeId);
    if (found) return found;
  }
  
  return null;
}


export function findNodeByTokenId(root: VisualNode, tokenId: number): VisualNode | null {
  if (root.token_id === tokenId) return root;
  
  for (const child of root.children) {
    const found = findNodeByTokenId(child, tokenId);
    if (found) return found;
  }
  
  return null;
}


export function findNodeByPath(root: VisualNode, path: string): VisualNode | null {
  if (root.path === path) return root;
  
  for (const child of root.children) {
    
    if (!path.startsWith(child.path)) continue;
    
    const found = findNodeByPath(child, path);
    if (found) return found;
  }
  
  return null;
}


export function findParentNode(root: VisualNode, nodeId: string): VisualNode | null {
  
  for (const child of root.children) {
    if (child.id === nodeId) return root;
  }
  
  
  for (const child of root.children) {
    const parent = findParentNode(child, nodeId);
    if (parent) return parent;
  }
  
  return null;
}


export function findParentNodeByTokenId(root: VisualNode, tokenId: number): VisualNode | null {
  
  for (const child of root.children) {
    if (child.token_id === tokenId) return root;
  }
  
  
  for (const child of root.children) {
    const parent = findParentNodeByTokenId(child, tokenId);
    if (parent) return parent;
  }
  
  return null;
}


export function findSiblingNodes(root: VisualNode, nodeId: string): VisualNode[] {
  const parent = findParentNode(root, nodeId);
  if (!parent) return [];
  
  
  return parent.children.filter(node => node.id !== nodeId);
}


export function getNodeIdFromSequence(
  completedSequences: CompletedSequence[], 
  sequenceId: string, 
  tokenIndex: number
): string | null {
  const sequence = completedSequences.find(seq => seq.id === sequenceId);
  if (!sequence || tokenIndex >= sequence.tokens.length) return null;
  
  return sequence.tokens[tokenIndex].nodeId;
}


export function getAllChildren(node: VisualNode): VisualNode[] {
  return node.children;
}


export function findAlternativeTokens(
  tree: VisualNode,
  completedSequences: CompletedSequence[],
  sequenceId: string,
  tokenIndex: number,
  tempSequence?: CompletedSequence | null
): Array<{ token: string; probability: number; nodeId: string }> {
  
  let nodeId: string | null = null;
  
  
  if (tempSequence && tempSequence.id === sequenceId) {
    if (tokenIndex >= 0 && tokenIndex < tempSequence.tokens.length) {
      nodeId = tempSequence.tokens[tokenIndex].nodeId;
    }
  } else {
    
    nodeId = getNodeIdFromSequence(completedSequences, sequenceId, tokenIndex);
  }
  
  if (!nodeId) return [];
  
  
  const selectedNode = findNodeById(tree, nodeId);
  if (!selectedNode) return [];
  
  
  const parentNode = findParentNode(tree, nodeId);
  if (!parentNode) return [];
  
  
  return parentNode.children.map(node => ({
    token: node.token,
    probability: node.prob,
    nodeId: node.id
  })).sort((a, b) => b.probability - a.probability); 
}


export function mergeTrees(prevTree: VisualNode | null, newTree: VisualNode, isSubtree: boolean = false): VisualNode {
  
  if (!prevTree) return newTree;
  
  
  if (!isSubtree) {
    return {
      ...prevTree,
      children: mergeChildNodes(prevTree.children, newTree.children)
    };
  }
  
  
  const targetNodeId = newTree.id;
  const targetNode = findNodeById(prevTree, targetNodeId);
  
  
  if (!targetNode) {
    return {
      ...prevTree,
      children: [...prevTree.children, newTree]
    };
  }
  
  
  return updateNodeInTree(prevTree, targetNode.id, {
    ...targetNode,
    children: mergeChildNodes(targetNode.children, newTree.children)
  });
}


export function mergeChildNodes(existingNodes: VisualNode[], incomingNodes: VisualNode[]): VisualNode[] {
  const result = [...existingNodes];
  
  for (const newNode of incomingNodes) {
    
    const existingNodeIndex = result.findIndex(node => node.id === newNode.id);
    
    if (existingNodeIndex === -1) {
      
      result.push(newNode);
    } else {
      
      const existingNode = result[existingNodeIndex];
      
      
      
      let mergedNodeState = newNode.nodeState;
      let mergedLastGenTime = newNode.lastGenerationRequestTime;
      
      if (existingNode.nodeState === 'completed' || 
         (existingNode.nodeState === 'generating' && newNode.nodeState !== 'completed')) {
        mergedNodeState = existingNode.nodeState;
        mergedLastGenTime = existingNode.lastGenerationRequestTime;
      }

      result[existingNodeIndex] = {
        ...existingNode,
        ...newNode,
        
        nodeState: mergedNodeState,
        lastGenerationRequestTime: mergedLastGenTime,
        
        isPinned: existingNode.isPinned,
        
        isEvaluated: existingNode.isEvaluated,
        evaluationCategory: existingNode.evaluationCategory,
        ancestorEvaluation: existingNode.ancestorEvaluation,
        
        cumulativeProb: newNode.cumulativeProb || existingNode.cumulativeProb,
        
        children: mergeChildNodes(existingNode.children, newNode.children)
      };
    }
  }
  
  
  return result.sort((a, b) => b.prob - a.prob);
}


export function updateNodeInTree(tree: VisualNode, nodeId: string, updatedNode: VisualNode): VisualNode {
  if (tree.id === nodeId) {
    return updatedNode;
  }
  
  return {
    ...tree,
    children: tree.children.map(child => 
      updateNodeInTree(child, nodeId, updatedNode)
    )
  };
}


export function getSelectedNodePath(
  completedSequences: CompletedSequence[],
  selectedToken: { sequenceId: string | null, tokenIndex: number | null }
): string[] {
  if (!selectedToken.sequenceId || selectedToken.tokenIndex === null) return [];
  
  const sequence = completedSequences.find(seq => seq.id === selectedToken.sequenceId);
  if (!sequence) return [];

  const selectedTokenInfo = sequence.tokens[selectedToken.tokenIndex];
  if (!selectedTokenInfo) return [];

  return sequence.tokens
    .slice(0, selectedToken.tokenIndex + 1)
    .map(t => t.nodeId);
}


export function updateGenerationStates(
  tree: VisualNode,
  nodeId: string,
  state: 'generating' | 'completed' | undefined,
  timestamp?: number
): VisualNode {
  
  if (tree.id === nodeId) {
    return {
      ...tree,
      nodeState: state,
      lastGenerationRequestTime: timestamp
    };
  }
  
  
  return {
    ...tree,
    children: tree.children.map(child => 
      updateGenerationStates(child, nodeId, state, timestamp)
    )
  };
}


export function findGeneratingNodes(root: VisualNode): string[] {
  const result: string[] = [];
  
  const traverse = (node: VisualNode) => {
    if (node.nodeState === 'generating') {
      result.push(node.id);
    }
    
    for (const child of node.children) {
      traverse(child);
    }
  };
  
  traverse(root);
  return result;
}


export function canGenerateFromNode(
  tree: VisualNode | null,
  nodeId: string,
  timeoutMs: number = 30000 
): boolean {
  if (!tree) return false;
  
  const node = findNodeById(tree, nodeId);
  if (!node) return false;
  
  
  if (node.nodeState === 'completed') return false;
  
  
  if (node.nodeState === 'generating' && node.lastGenerationRequestTime) {
    const elapsedTime = Date.now() - node.lastGenerationRequestTime;
    if (elapsedTime < timeoutMs) return false;
  }
  
  return true;
}


export function calculateProbabilityCoverage(tree: VisualNode, maxDepth: number = 10): number {
  if (!tree) return 0;
  
  
  const exploredProbabilityMass = calculateExploredProbabilityMass(tree, maxDepth);
  
  
  return exploredProbabilityMass * 100; 
}

function calculateExploredProbabilityMass(
  node: VisualNode, 
  maxDepth: number, 
  currentDepth: number = 0, 
  pathProb: number = 1.0
): number {
  
  if (currentDepth >= maxDepth || !node.children || node.children.length === 0) {
    return pathProb;
  }
  
  
  let totalMass = 0;
  for (const child of node.children) {
    totalMass += calculateExploredProbabilityMass(
      child, 
      maxDepth, 
      currentDepth + 1, 
      pathProb * child.prob
    );
  }
  
  return totalMass;
}


export function calculateDepthWiseCoverage(tree: VisualNode, maxDepth: number = 10): {
  coverageByDepth: number[];
  nodeCounts: number[];
} {
  if (!tree) return { coverageByDepth: [], nodeCounts: [] };
  
  const coverageByDepth: number[] = [];
  const nodeCounts: number[] = [];
  
  for (let depth = 1; depth <= maxDepth; depth++) {
    const result = getNodesAndProbabilityAtDepth(tree, depth);
    coverageByDepth.push(result.totalProb * 100); 
    nodeCounts.push(result.nodeCount);
  }
  
  return { coverageByDepth, nodeCounts };
}


export function generateSequenceFromNode(
  tree: VisualNode,
  nodeId: string,
  maxLength: number = 100
): CompletedSequence | null {
  const startNode = findNodeById(tree, nodeId);
  if (!startNode) return null;

  
  const EOT_TOKENS = ['<|eot_id|>', '</s>', '<|endoftext|>'];
  
  
  const pathToRoot: VisualNode[] = [];
  let currentNode: VisualNode | null = startNode;
  
  while (currentNode) {
    pathToRoot.unshift(currentNode);
    if (currentNode.id === 'root') break;
    currentNode = findParentNode(tree, currentNode.id);
  }
  
  
  const pathFromNode: VisualNode[] = [];
  currentNode = startNode;
  let steps = 0;
  
  while (currentNode && steps < maxLength) {
    
    if (!currentNode.children || currentNode.children.length === 0 || 
        EOT_TOKENS.includes(currentNode.token)) {
      break;
    }
    
    
    const bestChild: VisualNode = currentNode.children.reduce((best: VisualNode, child: VisualNode) => 
      child.prob > best.prob ? child : best
    );
    
    pathFromNode.push(bestChild);
    currentNode = bestChild;
    steps++;
  }
  
  
  
  const fullPath = [...pathToRoot, ...pathFromNode];
  
  
  const tokens: TokenInfo[] = [];
  let text = '';
  let totalProb = 1;
  
  for (const node of fullPath) {
    if (node.id === 'root' && node.token === 'root') continue;
    
    const startIndex = text.length;
    text += node.token;
    
    tokens.push({
      token: node.token,
      nodeId: node.id,
      prob: node.prob,
      startIndex,
      endIndex: startIndex + node.token.length
    });
    
    totalProb *= node.prob;
  }
  
  return {
    id: `seq-from-${nodeId}`,
    text,
    tokens,
    totalProb,
    path: text
  };
}


function getNodesAndProbabilityAtDepth(
  root: VisualNode, 
  targetDepth: number
): { totalProb: number; nodeCount: number } {
  const nodesAtDepth: { node: VisualNode; pathProb: number }[] = [];
  
  
  const EOT_TOKENS = ['<|eot_id|>', '</s>', '<|endoftext|>'];
  
  function collectNodesAtDepth(
    node: VisualNode, 
    currentDepth: number = 0, 
    pathProb: number = 1.0
  ) {
    
    if (currentDepth === targetDepth) {
      nodesAtDepth.push({ node, pathProb });
      return;
    }
    
    
    if (!node.children || node.children.length === 0) {
      
      if (EOT_TOKENS.includes(node.token) && currentDepth < targetDepth) {
        nodesAtDepth.push({ node, pathProb });
      }
      return;
    }
    
    
    for (const child of node.children) {
      collectNodesAtDepth(child, currentDepth + 1, pathProb * child.prob);
    }
  }
  
  collectNodesAtDepth(root);
  
  
  const totalProb = nodesAtDepth.reduce((sum, item) => sum + item.pathProb, 0);
  
  return { 
    totalProb, 
    nodeCount: nodesAtDepth.length
  };
}