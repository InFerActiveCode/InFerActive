import { VisualNode, CompletedSequence, TokenInfo } from '../types/types';

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
    const nodeId = root.id;
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

        if ((!childNode.children || childNode.children.length === 0) &&
            childNode.good !== undefined) {
          visualNode.evaluationCategory = childNode.good === 1 ? 'good' : 'bad';
          visualNode.isEvaluated = true;
        }

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

      // 부모 트리에서 상위 노드들 찾기
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

// 깊이별 확률 커버리지 계산 함수
export function calculateDepthWiseCoverage(tree: VisualNode, maxDepth: number = 10): {
  coverageByDepth: number[];
  nodeCounts: number[];
} {
  if (!tree) return { coverageByDepth: [], nodeCounts: [] };

  const coverageByDepth: number[] = [];
  const nodeCounts: number[] = [];

  for (let depth = 1; depth <= maxDepth; depth++) {
    const result = getNodesAndProbabilityAtDepth(tree, depth);
    coverageByDepth.push(result.totalProb * 100); // 퍼센트로 변환
    nodeCounts.push(result.nodeCount);
  }

  return { coverageByDepth, nodeCounts };
}

// 노드에서 최고 확률 경로를 따라가서 시퀀스 생성
export function generateSequenceFromNode(
  tree: VisualNode,
  nodeId: string,
  maxLength: number = 100
): CompletedSequence | null {
  const startNode = findNodeById(tree, nodeId);
  if (!startNode) return null;

  // EOT 토큰 목록
  const EOT_TOKENS = ['<|eot_id|>', '</s>', '<|endoftext|>'];

  // 시작 노드부터 루트까지의 경로 구하기
  const pathToRoot: VisualNode[] = [];
  let currentNode: VisualNode | null = startNode;

  while (currentNode) {
    pathToRoot.unshift(currentNode);
    if (currentNode.id === 'root') break;
    currentNode = findParentNode(tree, currentNode.id);
  }

  // 시작 노드부터 최고 확률 자식을 따라가기
  const pathFromNode: VisualNode[] = [];
  currentNode = startNode;
  let steps = 0;

  while (currentNode && steps < maxLength) {
    // 자식이 없거나 EOT 토큰이면 종료
    if (!currentNode.children || currentNode.children.length === 0 ||
        EOT_TOKENS.includes(currentNode.token)) {
      break;
    }

    // 최고 확률 자식 찾기
    const bestChild: VisualNode = currentNode.children.reduce((best: VisualNode, child: VisualNode) =>
      child.prob > best.prob ? child : best
    );

    pathFromNode.push(bestChild);
    currentNode = bestChild;
    steps++;
  }

  // 전체 경로 = 루트에서 시작 노드까지 + 시작 노드에서 끝까지
  // pathToRoot는 이미 시작 노드를 포함하고 있으므로 그대로 사용
  const fullPath = [...pathToRoot, ...pathFromNode];

  // 시퀀스 생성
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

// 특정 깊이의 노드들과 총 확률 계산
function getNodesAndProbabilityAtDepth(
  root: VisualNode,
  targetDepth: number
): { totalProb: number; nodeCount: number } {
  const nodesAtDepth: { node: VisualNode; pathProb: number }[] = [];

  // EOT 토큰 목록 정의
  const EOT_TOKENS = ['<|eot_id|>', '</s>', '<|endoftext|>'];

  function collectNodesAtDepth(
    node: VisualNode,
    currentDepth: number = 0,
    pathProb: number = 1.0
  ) {
    // 목표 깊이에 도달하면 노드 저장
    if (currentDepth === targetDepth) {
      nodesAtDepth.push({ node, pathProb });
      return;
    }

    // 리프 노드인 경우
    if (!node.children || node.children.length === 0) {
      // EOT 토큰인 경우에만 확률에 기여
      if (EOT_TOKENS.includes(node.token) && currentDepth < targetDepth) {
        nodesAtDepth.push({ node, pathProb });
      }
      return;
    }

    // 자식 노드 재귀 탐색
    for (const child of node.children) {
      collectNodesAtDepth(child, currentDepth + 1, pathProb * child.prob);
    }
  }

  collectNodesAtDepth(root);

  // 총 확률 계산 (모든 경로 확률의 합)
  const totalProb = nodesAtDepth.reduce((sum, item) => sum + item.pathProb, 0);

  return {
    totalProb,
    nodeCount: nodesAtDepth.length
  };
}

/**
 * 트리의 모든 leaf node를 수집하여 CompletedSequence 배열로 반환
 * - DFS로 순회하며 parentMap을 구축 (O(n))
 * - 각 leaf에서 root까지 역추적하여 시퀀스 생성
 * - totalProb 기준 내림차순 정렬
 */
export function collectLeafSequences(root: VisualNode): CompletedSequence[] {
  if (!root) return [];

  const EOT_TOKENS = ['<|eot_id|>', '</s>', '<|endoftext|>'];

  // 1. DFS로 parentMap 구축 + leaf 수집
  const parentMap = new Map<string, VisualNode | null>();
  const leaves: VisualNode[] = [];
  parentMap.set(root.id, null);

  const stack: Array<{ node: VisualNode; parent: VisualNode | null }> = [{ node: root, parent: null }];
  while (stack.length > 0) {
    const { node, parent } = stack.pop()!;
    parentMap.set(node.id, parent);

    if (node.children.length === 0) {
      leaves.push(node);
    } else {
      for (const child of node.children) {
        stack.push({ node: child, parent: node });
      }
    }
  }

  // 2. 각 leaf에서 root까지 역추적하여 시퀀스 생성
  const sequences: CompletedSequence[] = [];

  for (const leaf of leaves) {
    const chain: VisualNode[] = [];
    let current: VisualNode | null | undefined = leaf;

    while (current) {
      chain.unshift(current);
      current = parentMap.get(current.id);
    }

    const tokens: TokenInfo[] = [];
    let text = '';
    let totalProb = 1;

    for (const node of chain) {
      if (node.id === root.id && node.token === 'root') continue;
      // EOT 토큰은 텍스트에서 제외하되 확률에는 포함
      if (EOT_TOKENS.includes(node.token)) {
        totalProb *= node.prob;
        continue;
      }

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

    sequences.push({
      id: `leaf-${leaf.id}`,
      text,
      tokens,
      totalProb,
      path: text
    });
  }

  // 3. totalProb 내림차순 정렬
  sequences.sort((a, b) => b.totalProb - a.totalProb);

  return sequences;
}

/**
 * JSON 로드 시 리프 노드의 evaluationCategory를 기반으로 재귀적 통합 수행
 * - 단일 경로 통합: 부모가 단일 자식만 가지면 최상위로 마킹 이동
 * - 형제 통합: 모든 형제가 같은 카테고리면 부모로 통합
 * - ancestorEvaluation 전파: 마킹된 노드의 자손들에 상속
 */
export function applyInitialEvaluations(
  tree: VisualNode,
  evaluatedNodes: { good: Set<string>; bad: Set<string> }
): void {
  // 1단계: 리프 노드들에서 evaluationCategory가 설정된 노드들을 evaluatedNodes에 수집
  collectEvaluatedNodes(tree, evaluatedNodes);

  // 2단계: 각 마킹된 노드에 대해 단일 경로 통합 수행
  const nodesToProcess = [...Array.from(evaluatedNodes.good), ...Array.from(evaluatedNodes.bad)];
  for (const nodeId of nodesToProcess) {
    const node = findNodeById(tree, nodeId);
    if (!node || !node.evaluationCategory) continue;

    const category = node.evaluationCategory;
    consolidateSinglePath(tree, node, category, evaluatedNodes);
  }

  // 3단계: 형제 통합 수행 (리프부터 루트 방향으로)
  consolidateSiblingsRecursive(tree, tree, evaluatedNodes);

  // 4단계: 마킹된 노드들의 자손에 ancestorEvaluation 전파
  propagateAncestorEvaluation(tree, null);
}

/**
 * 트리를 순회하며 evaluationCategory가 설정된 노드들을 수집
 */
function collectEvaluatedNodes(
  node: VisualNode,
  evaluatedNodes: { good: Set<string>; bad: Set<string> }
): void {
  if (node.evaluationCategory === 'good') {
    evaluatedNodes.good.add(node.id);
  } else if (node.evaluationCategory === 'bad') {
    evaluatedNodes.bad.add(node.id);
  }

  for (const child of node.children) {
    collectEvaluatedNodes(child, evaluatedNodes);
  }
}

/**
 * 단일 경로 통합: 부모가 단일 자식만 가지면 마킹을 최상위로 이동
 */
function consolidateSinglePath(
  tree: VisualNode,
  node: VisualNode,
  category: 'good' | 'bad',
  evaluatedNodes: { good: Set<string>; bad: Set<string> }
): void {
  let currentNodeId = node.id;
  let topSingleParentId: string | null = null;

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
      node.evaluationCategory = null;
      node.isEvaluated = false;
      evaluatedNodes[category].delete(node.id);

      topParent.evaluationCategory = category;
      topParent.isEvaluated = true;
      evaluatedNodes[category].add(topSingleParentId);
    }
  }
}

/**
 * 형제 통합: 모든 형제가 같은 카테고리면 부모로 통합 (재귀적)
 */
function consolidateSiblingsRecursive(
  rootTree: VisualNode,
  currentNode: VisualNode,
  evaluatedNodes: { good: Set<string>; bad: Set<string> }
): void {
  // 후위 순회로 리프부터 처리
  for (const child of currentNode.children) {
    consolidateSiblingsRecursive(rootTree, child, evaluatedNodes);
  }

  // 현재 노드가 루트면 스킵
  if (currentNode.id === 'root') return;

  // 현재 노드의 부모에서 형제 통합 체크
  checkAndConsolidateAtNode(rootTree, currentNode, evaluatedNodes);
}

/**
 * 특정 노드에서 형제 통합 체크 및 상위로 재귀 전파
 */
function checkAndConsolidateAtNode(
  rootTree: VisualNode,
  node: VisualNode,
  evaluatedNodes: { good: Set<string>; bad: Set<string> }
): void {
  const parent = findParentNode(rootTree, node.id);
  if (!parent || parent.id === 'root') return;

  // 모든 형제가 같은 카테고리인지 확인
  const siblings = parent.children;
  if (siblings.length <= 1) return;

  const firstCategory = siblings[0].evaluationCategory;
  if (!firstCategory) return;

  const allSameCategory = siblings.every(
    sibling => sibling.evaluationCategory === firstCategory
  );

  if (allSameCategory) {
    // 모든 형제의 마크 제거
    for (const sibling of siblings) {
      sibling.evaluationCategory = null;
      sibling.isEvaluated = false;
      evaluatedNodes[firstCategory].delete(sibling.id);
    }

    // 부모에 마킹
    parent.evaluationCategory = firstCategory;
    parent.isEvaluated = true;
    evaluatedNodes[firstCategory].add(parent.id);

    // 부모가 마킹되었으니 단일 경로 통합 체크
    consolidateSinglePath(rootTree, parent, firstCategory, evaluatedNodes);

    // 부모가 마킹되었으니 상위 형제 통합도 재귀적으로 체크
    checkAndConsolidateAtNode(rootTree, parent, evaluatedNodes);
  }
}

/**
 * 마킹된 노드의 자손들에 ancestorEvaluation 전파
 */
function propagateAncestorEvaluation(
  node: VisualNode,
  parentEvaluation: 'good' | 'bad' | null
): void {
  // 현재 노드에 ancestorEvaluation 설정 (자신이 마킹되지 않은 경우만)
  if (!node.evaluationCategory && parentEvaluation) {
    node.ancestorEvaluation = parentEvaluation;
  }

  // 현재 노드가 마킹되어 있으면 자손들에 그 값을 전파
  const evaluationToPropagate = node.evaluationCategory || parentEvaluation;

  for (const child of node.children) {
    propagateAncestorEvaluation(child, evaluationToPropagate);
  }
}
