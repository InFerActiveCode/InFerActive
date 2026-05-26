import * as d3 from 'd3';
import { flextree } from 'd3-flextree';
import leftAlignedFlextree from '../utils/leftAlignedFlextree';
import { VisualNode, CompletedSequence, TokenSelection } from '../types/types';
import { findNodeById } from '../utils/treeTransform';

export interface EvaluationFilters {
  showGood: boolean;
  showBad: boolean;
  showUnmarked: boolean;
}

interface EvaluationFlags {
  isGood: boolean;
  isBad: boolean;
  isUnmarked: boolean;
}

const getEvaluationFlags = (
  node: Pick<VisualNode, 'evaluationCategory' | 'ancestorEvaluation'>
): EvaluationFlags => {
  const isGood = node.evaluationCategory === 'good' || node.ancestorEvaluation === 'good';
  const isBad = node.evaluationCategory === 'bad' || node.ancestorEvaluation === 'bad';
  const isUnmarked = !node.evaluationCategory && !node.ancestorEvaluation;

  return { isGood, isBad, isUnmarked };
};

export const passesEvaluationFilter = (
  node: Pick<VisualNode, 'evaluationCategory' | 'ancestorEvaluation'>,
  filters?: EvaluationFilters
): boolean => {
  if (!filters) return true;

  const { isGood, isBad, isUnmarked } = getEvaluationFlags(node);

  return !(
    (isGood && !filters.showGood) ||
    (isBad && !filters.showBad) ||
    (isUnmarked && !filters.showUnmarked)
  );
};

export interface ExtendedVisualNode extends VisualNode {
  isPinned?: boolean;
  isGenerating?: boolean;
  isCompleted?: boolean;
}

export type HierarchyPointNodeWithData = d3.HierarchyPointNode<ExtendedVisualNode> & {
  nodeWidth?: number;
};

let _cachedCtx: CanvasRenderingContext2D | null = null;
const _isWindows = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes('win');

function getCanvasContext(): CanvasRenderingContext2D | null {
  if (!_cachedCtx) {
    const canvas = document.createElement('canvas');
    _cachedCtx = canvas.getContext('2d');
    if (_cachedCtx) _cachedCtx.font = '12px monospace';
  }
  return _cachedCtx;
}

// 텍스트 너비 계산 함수
export const calculateTextWidth = (text: string): number => {
  const minWidth = 5;
  if (!text) return minWidth;

  const context = getCanvasContext();
  if (context) {
    let width = Math.ceil(context.measureText(text).width);
    if (_isWindows) {
      width = Math.ceil(width * 1.15);
    }
    return Math.max(minWidth, width);
  }

  return Math.max(minWidth, text.length * 7);
};

const WORD_SPLIT_REGEX = /[ \n]+/;

export const DEFAULT_NODE_MAX_WIDTH = 600;
export const MERGED_NODE_MAX_WIDTH = 500;
export const ROOT_NODE_MAX_WIDTH_RATIO = 1;
const ROOT_ICON_REFERENCE_TEXT = 'ro';
export const NODE_MAX_LINES = 1;
export const NODE_EXPANDED_MAX_LINES = 5;
export const LEAF_NODE_EXPANDED_MAX_LINES = 3;
export const NODE_LINE_HEIGHT = 16;
export const NODE_VERTICAL_PADDING = 12;
export const OVERVIEW_FILTERED_NODE_RADIUS = 2;
export const OVERVIEW_FILTERED_NODE_DIAMETER = OVERVIEW_FILTERED_NODE_RADIUS * 2;
export const OVERVIEW_FILTERED_NODE_LAYOUT_HEIGHT = OVERVIEW_FILTERED_NODE_DIAMETER + 1;
export const OVERVIEW_FILTERED_NODE_LAYOUT_WIDTH = 100;
export const OVERVIEW_NODE_PADDING = 6;

export interface WrappedWordFragment {
  text: string;
  tokenIndex?: number;
}

export interface WrappedTextLayout {
  lines: WrappedWordFragment[][];
  lineWidths: number[];
  totalLineCount: number;
  visibleLineCount: number;
  truncated: boolean;
  maxVisibleLineWidth: number;
  height: number;
}

type NodeTextSource = {
  token?: string;
  isTextExpanded?: boolean;
  forcedTextMaxLines?: number;
  mergedNodes?: {
    tokens?: string[];
  };
};

const buildWordFragmentsFromToken = (
  token: string,
  tokenIndex?: number
): WrappedWordFragment[] => {
  const words = token.split(WORD_SPLIT_REGEX);
  return words.map((word, idx) => ({
    text: word + (idx < words.length - 1 ? ' ' : ''),
    tokenIndex
  }));
};

export const buildNodeWordFragments = (
  nodeData: NodeTextSource
): WrappedWordFragment[] => {
  if (nodeData.mergedNodes?.tokens && nodeData.mergedNodes.tokens.length > 0) {
    const mergedFragments: WrappedWordFragment[] = [];
    nodeData.mergedNodes.tokens.forEach((token, tokenIndex) => {
      mergedFragments.push(...buildWordFragmentsFromToken(token, tokenIndex));
    });
    return mergedFragments.length > 0 ? mergedFragments : [{ text: '' }];
  }

  return buildWordFragmentsFromToken(nodeData.token || '');
};

export const getNodeMaxTextWidth = (nodeData: NodeTextSource): number => {
  return nodeData.mergedNodes ? MERGED_NODE_MAX_WIDTH : DEFAULT_NODE_MAX_WIDTH;
};

interface NodeTextLayoutOptions {
  isRootNode?: boolean;
  isMergedPrefixUnderRoot?: boolean;
  isLeafNode?: boolean;
}

export const calculateWrappedTextLayout = (
  fragments: WrappedWordFragment[],
  maxWidth: number,
  maxLines: number = Number.MAX_SAFE_INTEGER
): WrappedTextLayout => {
  const wrapWidth = Math.max(1, maxWidth - 8);
  const normalizedFragments = fragments.length > 0 ? fragments : [{ text: '' }];
  const expandedFragments = normalizedFragments.flatMap((fragment) => {
    if (calculateTextWidth(fragment.text) <= wrapWidth) {
      return [fragment];
    }

    const chars = Array.from(fragment.text);
    if (chars.length <= 1) {
      return [fragment];
    }

    const chunked: WrappedWordFragment[] = [];
    let currentChunk = '';

    for (const ch of chars) {
      const nextChunk = currentChunk + ch;
      if (currentChunk && calculateTextWidth(nextChunk) > wrapWidth) {
        chunked.push({ text: currentChunk, tokenIndex: fragment.tokenIndex });
        currentChunk = ch;
      } else {
        currentChunk = nextChunk;
      }
    }

    if (currentChunk) {
      chunked.push({ text: currentChunk, tokenIndex: fragment.tokenIndex });
    }

    return chunked.length > 0 ? chunked : [fragment];
  });
  const wrappedLines: WrappedWordFragment[][] = [];
  const lineWidths: number[] = [];
  let currentLine: WrappedWordFragment[] = [];
  let currentLineWidth = 0;

  expandedFragments.forEach((fragment) => {
    const fragmentWidth = calculateTextWidth(fragment.text);

    if (currentLineWidth + fragmentWidth > wrapWidth && currentLine.length > 0) {
      wrappedLines.push(currentLine);
      lineWidths.push(currentLineWidth);
      currentLine = [fragment];
      currentLineWidth = fragmentWidth;
    } else {
      currentLine.push(fragment);
      currentLineWidth += fragmentWidth;
    }
  });

  if (currentLine.length === 0) {
    currentLine = [{ text: '' }];
    currentLineWidth = calculateTextWidth('');
  }

  wrappedLines.push(currentLine);
  lineWidths.push(currentLineWidth);

  const normalizedMaxLines = Math.max(1, maxLines);
  const visibleLines = wrappedLines.slice(0, normalizedMaxLines);
  const visibleLineWidths = lineWidths.slice(0, normalizedMaxLines);
  const visibleLineCount = visibleLines.length;

  return {
    lines: visibleLines,
    lineWidths: visibleLineWidths,
    totalLineCount: wrappedLines.length,
    visibleLineCount,
    truncated: wrappedLines.length > normalizedMaxLines,
    maxVisibleLineWidth: Math.max(...visibleLineWidths, 0),
    height: visibleLineCount * NODE_LINE_HEIGHT + NODE_VERTICAL_PADDING
  };
};

export const calculateNodeTextLayout = (
  nodeData: NodeTextSource,
  maxLines?: number,
  options?: NodeTextLayoutOptions
): WrappedTextLayout & { maxWidth: number } => {
  const expandedMaxLines = options?.isLeafNode
    ? LEAF_NODE_EXPANDED_MAX_LINES
    : NODE_EXPANDED_MAX_LINES;
  const resolvedMaxLines = maxLines ?? nodeData.forcedTextMaxLines ?? (
    nodeData.isTextExpanded ? expandedMaxLines : NODE_MAX_LINES
  );
  const baseMaxWidth = getNodeMaxTextWidth(nodeData);
  const shouldApplyRootWidthRatio = Boolean(options?.isRootNode || options?.isMergedPrefixUnderRoot);
  const maxWidth = shouldApplyRootWidthRatio
    ? Math.max(1, Math.floor(baseMaxWidth * ROOT_NODE_MAX_WIDTH_RATIO))
    : baseMaxWidth;
  const fragments = buildNodeWordFragments(nodeData);
  const layout = calculateWrappedTextLayout(fragments, maxWidth, resolvedMaxLines);
  if (options?.isRootNode && nodeData.token === 'root') {
    return {
      ...layout,
      maxWidth,
      maxVisibleLineWidth: calculateTextWidth(ROOT_ICON_REFERENCE_TEXT)
    };
  }
  return { ...layout, maxWidth };
};

export const calculateTextWrapping = (text: string, maxWidth: number = 500): { lines: string[], height: number } => {
  const fragments = buildWordFragmentsFromToken(text || '');
  const wrapping = calculateWrappedTextLayout(fragments, maxWidth, Number.MAX_SAFE_INTEGER);
  return {
    lines: wrapping.lines.map(line => line.map(fragment => fragment.text).join('')),
    height: wrapping.height
  };
};

// 트리 차원 계산 함수
export const calculateTreeDimensions = (
  root: d3.HierarchyNode<VisualNode>,
  containerWidth: number = 800,
  containerHeight: number = 600
): { width: number, height: number } => {
  const maxDepth = d3.max(root.descendants(), d => d.depth) || 0;
  const leafNodes = root.leaves().length;

  const optimalBranchLength = 60;
  const calculatedWidth = (maxDepth + 1) * optimalBranchLength * 2;
  const minVerticalSpacing = 40;
  const calculatedHeight = Math.max(
    containerHeight,
    leafNodes * minVerticalSpacing
  );

  return {
    width: Math.max(calculatedWidth, containerWidth),
    height: calculatedHeight
  };
};

// 선택된 토큰 경로 계산
export const getSelectedNodePath = (
  completedSequences: CompletedSequence[],
  selectedToken: TokenSelection
): string[] => {
  if (!selectedToken.sequenceId || selectedToken.tokenIndex === null) return [];

  const sequence = completedSequences.find(seq => seq.id === selectedToken.sequenceId);
  if (!sequence) return [];

  const selectedTokenInfo = sequence.tokens[selectedToken.tokenIndex];
  if (!selectedTokenInfo) return [];

  return sequence.tokens
    .slice(0, selectedToken.tokenIndex + 1)
    .map(t => t.nodeId);
};

// 루트에서 특정 노드까지의 경로 반환
export const findPathToNode = (
  root: VisualNode,
  targetId: string
): VisualNode[] => {
  const path: VisualNode[] = [];

  const dfs = (current: VisualNode): boolean => {
    path.push(current);
    if (current.id === targetId) return true;

    for (const child of current.children) {
      if (dfs(child)) return true;
    }

    path.pop();
    return false;
  };

  return dfs(root) ? path : [];
};

export const compareNodesByProbability = (
  a: Pick<VisualNode, 'id' | 'prob'>,
  b: Pick<VisualNode, 'id' | 'prob'>
): number => {
  const probDiff = (b.prob || 0) - (a.prob || 0);
  if (probDiff !== 0) return probDiff;
  return a.id.localeCompare(b.id);
};

// Top-N 유사 선택: 부모 간 우선순위는 frontier(BFS) 순서를 유지하고,
// 같은 부모의 자식들은 호출자가 넘긴 순서를 그대로 우선순위로 사용한다.
export const selectTopNLikeLeafNodes = (
  root: VisualNode,
  target: number,
  getValidChildren: (node: VisualNode) => VisualNode[]
): VisualNode[] => {
  if (target <= 0) return [];

  const depth1Children = getValidChildren(root);
  if (depth1Children.length === 0) return [];

  let visibleLeaves =
    depth1Children.length >= target
      ? depth1Children.slice(0, target)
      : [...depth1Children];

  while (visibleLeaves.length < target) {
    const addedChildren = new Map<string, VisualNode[]>();
    let currentLeafCount = visibleLeaves.length;
    let hasExpandableLeaf = false;

    for (const leaf of visibleLeaves) {
      if (currentLeafCount >= target) break;

      const children = getValidChildren(leaf);
      if (children.length === 0) continue;

      hasExpandableLeaf = true;
      const selectableCount = Math.min(
        children.length,
        target - currentLeafCount + 1
      );

      if (selectableCount > 0) {
        addedChildren.set(leaf.id, children.slice(0, selectableCount));
        currentLeafCount += selectableCount - 1;
      }
    }

    if (!hasExpandableLeaf || addedChildren.size === 0) {
      break;
    }

    const nextVisibleLeaves: VisualNode[] = [];
    for (const leaf of visibleLeaves) {
      const selected = addedChildren.get(leaf.id);
      if (selected) {
        nextVisibleLeaves.push(...selected);
      } else {
        nextVisibleLeaves.push(leaf);
      }
    }

    visibleLeaves = nextVisibleLeaves;
  }

  return visibleLeaves.map((leaf) => {
    let current = leaf;
    while (true) {
      const validChildren = getValidChildren(current);
      if (validChildren.length === 0) break;
      current = validChildren[0];
    }
    return current;
  });
};

export const findRightmostSinglePathNode = (
  tree: VisualNode,
  subtreeRootId: string
): VisualNode | null => {
  const subtreeRoot = findNodeById(tree, subtreeRootId);
  if (!subtreeRoot) return null;

  let current = subtreeRoot;
  while (current.children.length === 1) {
    current = current.children[0];
  }

  return current;
};

// Top-N 전용: 선택 노드를 서브트리 루트로 승격한 가상 루트 생성
export const createTopNSubtreeRoot = (
  tree: VisualNode,
  subtreeRootId: string
): VisualNode | null => {
  if (subtreeRootId === tree.id) {
    return tree;
  }

  const subtreeRoot = findNodeById(tree, subtreeRootId);
  if (!subtreeRoot) return null;

  const effectiveSubtreeRoot = findRightmostSinglePathNode(tree, subtreeRootId);
  if (!effectiveSubtreeRoot) return null;

  const wasAutoAdvanced = effectiveSubtreeRoot.id !== subtreeRoot.id;

  const pathToSubtreeRoot = findPathToNode(tree, effectiveSubtreeRoot.id);
  if (pathToSubtreeRoot.length === 0) return null;

  // 루트 노드는 항상 depth 0에서 독립적으로 유지한다.
  // 경로의 루트 이후(prefix)만 병합된 단일 자식으로 표현한다.
  const [rootNode, ...prefixPathNodes] = pathToSubtreeRoot;
  if (!rootNode) return null;
  if (prefixPathNodes.length === 0) return rootNode;

  const mergedTokens = prefixPathNodes.map(n => n.token);
  const mergedProbs = prefixPathNodes.map(n => n.prob);
  const mergedNodeIds = prefixPathNodes.map(n => n.id);
  const firstMergedNode = prefixPathNodes[0] || effectiveSubtreeRoot;

  const mergedPrefixNode: VisualNode = {
    ...effectiveSubtreeRoot,
    prob: firstMergedNode.prob,
    cumulativeProb: firstMergedNode.cumulativeProb,
    token: mergedTokens.join('') || effectiveSubtreeRoot.token,
    isTextExpanded: prefixPathNodes.length > 1 ||
      wasAutoAdvanced ||
      prefixPathNodes.some(n => Boolean(n.isTextExpanded)) ||
      Boolean(effectiveSubtreeRoot.isTextExpanded),
    mergedNodes: {
      tokens: mergedTokens.length > 0 ? mergedTokens : [effectiveSubtreeRoot.token],
      probs: mergedProbs.length > 0 ? mergedProbs : [effectiveSubtreeRoot.prob],
      nodeIds: mergedNodeIds.length > 0 ? mergedNodeIds : [effectiveSubtreeRoot.id]
    },
    children: effectiveSubtreeRoot.children
  };

  return {
    ...rootNode,
    mergedNodes: undefined,
    isTextExpanded: Boolean(rootNode.isTextExpanded),
    children: [mergedPrefixNode]
  };
};

const buildCenterOutSlots = (length: number): number[] => {
  if (length <= 0) return [];

  const slots: number[] = [];
  const leftCenter = Math.floor((length - 1) / 2);
  const rightCenter = Math.ceil((length - 1) / 2);

  slots.push(leftCenter);
  if (rightCenter !== leftCenter) {
    slots.push(rightCenter);
  }

  let offset = 1;
  while (slots.length < length) {
    const upper = leftCenter - offset;
    const lower = rightCenter + offset;

    if (upper >= 0) {
      slots.push(upper);
    }
    if (slots.length >= length) break;

    if (lower < length) {
      slots.push(lower);
    }

    offset += 1;
  }

  return slots;
};

export const orderChildrenByTopNRank = (
  children: VisualNode[],
  topNRankByNodeId?: ReadonlyMap<string, number>,
  centerTopNNodes: boolean = false
): VisualNode[] => {
  if (!centerTopNNodes || !topNRankByNodeId || children.length <= 1) {
    return children;
  }

  const rankedChildren = children.filter(child => topNRankByNodeId.has(child.id));
  if (rankedChildren.length === 0) {
    return children;
  }

  const prioritizedChildren = children
    .map((child, originalIndex) => ({
      child,
      originalIndex,
      rank: topNRankByNodeId.get(child.id) ?? Number.POSITIVE_INFINITY
    }))
    .sort((a, b) => {
      if (a.rank !== b.rank) {
        return a.rank - b.rank;
      }
      return a.originalIndex - b.originalIndex;
    })
    .map(entry => entry.child);

  const orderedChildren = new Array<VisualNode>(prioritizedChildren.length);
  const slots = buildCenterOutSlots(prioritizedChildren.length);

  prioritizedChildren.forEach((child, index) => {
    orderedChildren[slots[index]] = child;
  });

  return orderedChildren;
};

const shiftHierarchySubtree = (
  subtreeRoot: HierarchyPointNodeWithData,
  deltaX: number
): void => {
  if (deltaX === 0) return;
  subtreeRoot.each(node => {
    (node as HierarchyPointNodeWithData).x += deltaX;
  });
};

const getHierarchyNodeHeight = (
  node: HierarchyPointNodeWithData,
  overviewMode: boolean,
  isMergedPrefixUnderRootNode: (n: any) => boolean
): number => {
  if (overviewMode && node.data.isFiltered) {
    return OVERVIEW_FILTERED_NODE_LAYOUT_HEIGHT;
  }

  const textLayout = calculateNodeTextLayout(node.data, undefined, {
    isRootNode: node.depth === 0,
    isMergedPrefixUnderRoot: isMergedPrefixUnderRootNode(node),
    isLeafNode: !node.children || node.children.length === 0
  });
  return textLayout.height + 10;
};

const getHierarchySubtreeExtents = (
  subtreeRoot: HierarchyPointNodeWithData,
  overviewMode: boolean,
  isMergedPrefixUnderRootNode: (n: any) => boolean
): { minX: number; maxX: number } => {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;

  subtreeRoot.each(node => {
    const hierarchyNode = node as HierarchyPointNodeWithData;
    const halfHeight = getHierarchyNodeHeight(
      hierarchyNode,
      overviewMode,
      isMergedPrefixUnderRootNode
    ) / 2;
    minX = Math.min(minX, hierarchyNode.x - halfHeight);
    maxX = Math.max(maxX, hierarchyNode.x + halfHeight);
  });

  return { minX, maxX };
};

type ChildPlacementDirection = 'free' | 'up' | 'down';

const centerTopNSubtreesWithinBands = (
  root: HierarchyPointNodeWithData,
  topNRankByNodeId: ReadonlyMap<string, number>,
  overviewMode: boolean,
  nodePadding: number,
  isMergedPrefixUnderRootNode: (n: any) => boolean
): void => {
  if (topNRankByNodeId.size === 0) return;

  const subtreeGap = overviewMode
    ? Math.max(nodePadding, 4)
    : Math.max(nodePadding, 12);
  const childPlacementDirectionByNodeId = new Map<string, ChildPlacementDirection>();

  const getPrioritizedChildren = (
    hierarchyNode: HierarchyPointNodeWithData
  ): HierarchyPointNodeWithData[] => {
    const children = hierarchyNode.children as HierarchyPointNodeWithData[] | undefined;
    if (!children || children.length === 0) return [];

    return [...children].sort((a, b) => {
      const rankA = topNRankByNodeId.get(a.data.id) ?? Number.POSITIVE_INFINITY;
      const rankB = topNRankByNodeId.get(b.data.id) ?? Number.POSITIVE_INFINITY;
      if (rankA !== rankB) {
        return rankA - rankB;
      }
      return a.x - b.x;
    });
  };

  const assignChildPlacementDirections = (
    hierarchyNode: HierarchyPointNodeWithData,
    direction: ChildPlacementDirection
  ) => {
    childPlacementDirectionByNodeId.set(hierarchyNode.data.id, direction);

    const prioritizedChildren = getPrioritizedChildren(hierarchyNode);
    if (prioritizedChildren.length === 0) return;

    if (direction === 'up' || direction === 'down') {
      prioritizedChildren.forEach(child => {
        assignChildPlacementDirections(child, direction);
      });
      return;
    }

    const [anchorChild, ...remainingChildren] = prioritizedChildren;
    if (!anchorChild) return;

    if (remainingChildren.length === 0) {
      assignChildPlacementDirections(anchorChild, 'free');
      return;
    }

    let nextSide: ChildPlacementDirection = 'up';
    assignChildPlacementDirections(anchorChild, nextSide === 'up' ? 'down' : 'up');

    remainingChildren.forEach(child => {
      assignChildPlacementDirections(child, nextSide);
      nextSide = nextSide === 'up' ? 'down' : 'up';
    });
  };

  assignChildPlacementDirections(root, 'free');

  root.eachAfter(node => {
    const hierarchyNode = node as HierarchyPointNodeWithData;
    const prioritizedChildren = getPrioritizedChildren(hierarchyNode);
    if (prioritizedChildren.length <= 1) return;

    const placementDirection =
      childPlacementDirectionByNodeId.get(hierarchyNode.data.id) ?? 'free';
    const anchorChild = prioritizedChildren[0];
    if (!anchorChild) return;

    shiftHierarchySubtree(anchorChild, hierarchyNode.x - anchorChild.x);

    const anchorExtents = getHierarchySubtreeExtents(
      anchorChild,
      overviewMode,
      isMergedPrefixUnderRootNode
    );
    let upperBoundary = anchorExtents.minX - subtreeGap;
    let lowerBoundary = anchorExtents.maxX + subtreeGap;

    const placeChildAbove = (child: HierarchyPointNodeWithData) => {
      const extents = getHierarchySubtreeExtents(
        child,
        overviewMode,
        isMergedPrefixUnderRootNode
      );
      shiftHierarchySubtree(child, upperBoundary - extents.maxX);
      const nextExtents = getHierarchySubtreeExtents(
        child,
        overviewMode,
        isMergedPrefixUnderRootNode
      );
      upperBoundary = nextExtents.minX - subtreeGap;
    };

    const placeChildBelow = (child: HierarchyPointNodeWithData) => {
      const extents = getHierarchySubtreeExtents(
        child,
        overviewMode,
        isMergedPrefixUnderRootNode
      );
      shiftHierarchySubtree(child, lowerBoundary - extents.minX);
      const nextExtents = getHierarchySubtreeExtents(
        child,
        overviewMode,
        isMergedPrefixUnderRootNode
      );
      lowerBoundary = nextExtents.maxX + subtreeGap;
    };

    if (placementDirection === 'up') {
      prioritizedChildren.slice(1).forEach(placeChildAbove);
      return;
    }

    if (placementDirection === 'down') {
      prioritizedChildren.slice(1).forEach(placeChildBelow);
      return;
    }

    let placeAbove = true;
    prioritizedChildren.slice(1).forEach(child => {
      if (placeAbove) {
        placeChildAbove(child);
      } else {
        placeChildBelow(child);
      }
      placeAbove = !placeAbove;
    });
  });
};

// 필터링된 자식 노드 반환 함수
export const getVisibleChildren = (
  node: VisualNode,
  bigTokenEnabled: boolean = true,
  globalTopNNodes?: Set<string>, // 전역 Top-N 노드 Set (옵션)
  topNRankByNodeId?: ReadonlyMap<string, number>,
  centerTopNNodes: boolean = false
): VisualNode[] => {
  // 노드 자체가 접혀있으면 자식 노드를 표시하지 않음
  if (node.isFolded) {
    return [];
  }

  let children = orderChildrenByTopNRank(
    [...node.children],
    topNRankByNodeId,
    centerTopNNodes
  );

  // 가장 높은 확률의 자식 찾기 (greedy child)
  const greedyChild = children.length > 0 ?
    children.reduce((max, child) => child.prob > max.prob ? child : max) : null;

  // globalTopNNodes가 제공된 경우 전역 Top-N 필터링 사용
  if (globalTopNNodes) {
    // 전역 Top-N 노드에 포함되는지 확인
    for (const child of children) {
      const isInTopN = globalTopNNodes.has(child.id);

      // 현재 노드가 user로 펼치는 경우 Greedy child는 항상 펼치기
      if (node.isUserFolded === false && greedyChild && child.id === greedyChild.id) {
        child.isUserFolded = false;
      }

      // 사용자가 직접 접거나 펼친 상태가 있는 경우, 그 상태를 우선 적용
      if (child.isUserFolded !== undefined) {
        child.isFolded = child.isUserFolded;
      } else {
        child.isFolded = !isInTopN;
      }
    }
  } else {
    // globalTopNNodes가 없는 경우 모든 노드 표시 (폴딩 없음)
    for (const child of children) {
      // 사용자가 직접 접거나 펼친 상태가 있는 경우만 적용
      if (child.isUserFolded !== undefined) {
        child.isFolded = child.isUserFolded;
      } else {
        // Greedy child는 항상 펼치기
        if (greedyChild && child.id === greedyChild.id) {
          child.isFolded = false;
        } else {
          child.isFolded = false;
        }
      }
    }
  }

  // 접히지 않은 자식 노드만 필터링
  const visibleChildren = children.filter(child => !child.isFolded);
  const foldedSiblingCountByAnchorId = new Map<string, number>();
  const userFoldedSiblingCountByAnchorId = new Map<string, number>();

  // 숨겨진 형제는 "바로 왼쪽의 마지막 visible 형제"에게만 귀속한다.
  let lastVisibleSiblingId: string | null = null;
  for (const child of children) {
    if (child.isFolded) {
      if (!lastVisibleSiblingId) continue;

      foldedSiblingCountByAnchorId.set(
        lastVisibleSiblingId,
        (foldedSiblingCountByAnchorId.get(lastVisibleSiblingId) ?? 0) + 1
      );
      if (child.isUserFolded === true) {
        userFoldedSiblingCountByAnchorId.set(
          lastVisibleSiblingId,
          (userFoldedSiblingCountByAnchorId.get(lastVisibleSiblingId) ?? 0) + 1
        );
      }
      continue;
    }

    lastVisibleSiblingId = child.id;
  }

  // 각 자식에 대해 단일 경로 압축 시도
  const processedChildren = visibleChildren.map((child) => {
    const foldedAfter = foldedSiblingCountByAnchorId.get(child.id) ?? 0;
    const userFoldedAfter = userFoldedSiblingCountByAnchorId.get(child.id) ?? 0;

    const childWithFoldedInfo = {
      ...child,
      foldedSiblingCount: foldedAfter,
      userFoldedSiblingCount: userFoldedAfter
    };
    // 단일 경로 압축 시도
    let current = childWithFoldedInfo;
    const mergedTokens = [current.token];
    const mergedProbs = [current.prob];
    const mergedNodeIds = [current.id];
    let mergedIsTextExpanded = Boolean(current.isTextExpanded);
    const hiddenBranchPoints: Array<{
      nodeId: string;
      tokenIndex: number;
      hiddenCount: number;
    }> = [];
    let totalHiddenBranches = 0;

    const applyFoldStateAndCollectVisibleChildren = (candidateChildren: VisualNode[]): VisualNode[] => {
      const greedyCandidate = candidateChildren.length > 0
        ? candidateChildren.reduce((max, candidate) => candidate.prob > max.prob ? candidate : max)
        : null;

      for (const nextChild of candidateChildren) {
        // Newly unfolded sibling branches should propagate the same greedy-path visibility
        // during this merge pass, otherwise their descendants lag one level behind.
        if (current.isUserFolded === false && greedyCandidate && nextChild.id === greedyCandidate.id) {
          nextChild.isUserFolded = false;
        }

        if (nextChild.isUserFolded !== undefined) {
          nextChild.isFolded = nextChild.isUserFolded;
        } else {
          if (globalTopNNodes) {
            nextChild.isFolded = !globalTopNNodes.has(nextChild.id);
          } else {
            nextChild.isFolded = false;
          }
        }
      }

      return candidateChildren.filter(nextChild => !nextChild.isFolded);
    };

    const collectHiddenBranchPoint = (
      branchNode: VisualNode,
      tokenIndex: number,
      candidateChildren: VisualNode[],
      visibleAtLevel: VisualNode[]
    ) => {
      if (visibleAtLevel.length !== 1) return;

      const hiddenCount = candidateChildren.length - visibleAtLevel.length;
      if (hiddenCount <= 0) return;

      hiddenBranchPoints.push({
        nodeId: branchNode.id,
        tokenIndex,
        hiddenCount
      });
      totalHiddenBranches += hiddenCount;
    };

    // 계속해서 단일 경로를 따라 내려감
    let candidateChildren = current.children || [];
    let nextChildren = applyFoldStateAndCollectVisibleChildren(candidateChildren);
    collectHiddenBranchPoint(current, 0, candidateChildren, nextChildren);

    // 단일 경로일 경우 병합
    while (nextChildren.length === 1 && !nextChildren[0].isExpanded) {
      current = {
        ...nextChildren[0],
        foldedSiblingCount: foldedAfter,
        userFoldedSiblingCount: userFoldedAfter
      };
      mergedTokens.push(current.token);
      mergedProbs.push(current.prob);
      mergedNodeIds.push(current.id);
      mergedIsTextExpanded = mergedIsTextExpanded || Boolean(current.isTextExpanded);

      // 다음 레벨 확인
      candidateChildren = current.children || [];
      nextChildren = applyFoldStateAndCollectVisibleChildren(candidateChildren);
      collectHiddenBranchPoint(current, mergedTokens.length - 1, candidateChildren, nextChildren);
    }

    // 빅토큰이 활성화되고 최소 2개 이상 노드가 병합될 때만 처리
    if (bigTokenEnabled && mergedTokens.length > 1) {
      return {
        ...childWithFoldedInfo,
        token: mergedTokens.join(''),
        isTextExpanded: mergedIsTextExpanded,
        mergedNodes: {
          tokens: mergedTokens,
          probs: mergedProbs,
          nodeIds: mergedNodeIds,
          hiddenBranches: totalHiddenBranches > 0
            ? {
                totalHidden: totalHiddenBranches,
                points: hiddenBranchPoints
              }
            : undefined
        },
        children: current.children
      };
    }

    return childWithFoldedInfo;
  });

  return processedChildren;
};

// Overview 모드용 자식 노드 반환 함수 - folded 노드도 포함하되 isFiltered로 마킹
export const getVisibleChildrenForOverview = (
  node: VisualNode,
  bigTokenEnabled: boolean = true,
  globalTopNNodes?: Set<string>,
  topNRankByNodeId?: ReadonlyMap<string, number>,
  centerTopNNodes: boolean = false
): VisualNode[] => {
  // 모든 자식을 포함하되, folded 노드는 isFiltered로 마킹
  const children = orderChildrenByTopNRank(
    [...node.children],
    topNRankByNodeId,
    centerTopNNodes
  );

  // 가장 높은 확률의 자식 찾기 (greedy child) - getVisibleChildren 로직 참조
  const greedyChild = children.length > 0 ?
    children.reduce((max, child) => child.prob > max.prob ? child : max) : null;

  // 각 자식에 대해 단일 경로 압축 시도 (빅토큰 로직)
  const processedChildren = children.map((child) => {
    // globalTopNNodes 기반으로 필터링 여부 결정 (표시용)
    const isFiltered = globalTopNNodes ? !globalTopNNodes.has(child.id) : false;




    let shouldFilter = isFiltered
    if (node.isUserFolded === false) {
      // 유저가 직접 접은게 아니라면 현재 노드가 user로 펼치는 경우 Greedy child는 항상 펼치기
      if (child.isUserFolded !== true && greedyChild && child.id === greedyChild.id) {
        child.isUserFolded = false;
      }
    }
    // 사용자가 명시적으로 접거나 펼친 상태가 있으면 부모가 filtered 되지 않았을 때 그 상태를 우선 적용
    if (node.isFiltered === false && child.isUserFolded !== undefined) {
      shouldFilter = child.isUserFolded;
    }
    else {
      shouldFilter = isFiltered;
    }

    // 빅토큰 로직: 단일 경로 압축 시도 (모든 노드에 적용)
    let current = child;
    const mergedTokens = [current.token];
    const mergedProbs = [current.prob];
    const mergedNodeIds = [current.id];
    let mergedIsTextExpanded = Boolean(current.isTextExpanded);

    // 계속해서 단일 경로를 따라 내려감
    let nextChildren = current.children || [];

    // Overview에서는 모든 자식을 고려 (필터링 없음)
    // 단일 경로일 경우 병합
    while (nextChildren.length === 1 && !nextChildren[0].isExpanded) {
      current = nextChildren[0];
      mergedTokens.push(current.token);
      mergedProbs.push(current.prob);
      mergedNodeIds.push(current.id);
      mergedIsTextExpanded = mergedIsTextExpanded || Boolean(current.isTextExpanded);

      // 다음 레벨 확인
      nextChildren = current.children || [];
    }

    // 빅토큰이 활성화되고 최소 2개 이상 노드가 병합될 때만 처리
    if (bigTokenEnabled && mergedTokens.length > 1) {
      return {
        ...child,
        token: mergedTokens.join(''),
        isTextExpanded: mergedIsTextExpanded,
        mergedNodes: {
          tokens: mergedTokens,
          probs: mergedProbs,
          nodeIds: mergedNodeIds
        },
        children: current.children,
        isFiltered: shouldFilter,  // 필터링 상태 표시
        isFolded: false // overview에서는 폴딩 무시
      };
    }

    return {
      ...child,
      isFiltered: shouldFilter,  // 필터링 상태 표시
      isFolded: false // overview에서는 폴딩 무시
    };
  });

  return processedChildren;
};

// D3 계층 데이터 생성
export const createHierarchyData = (
  node: VisualNode | null,
  nodePadding: number,
  leftAligned: boolean = false,
  bigTokenEnabled: boolean = true,
  evaluationFilters?: {
    showGood: boolean;
    showBad: boolean;
    showUnmarked: boolean;
  },
  globalTopNNodes?: Set<string>, // 전역 Top-N 노드 Set (옵션)
  useStandardTree: boolean = false, // d3 기본 트리 사용 여부
  overviewMode: boolean = false, // Overview 모드 여부
  centerTopNNodes: boolean = false,
  topNRankByNodeId?: ReadonlyMap<string, number>,
  overviewMaxDepth: number | null = null,
  overviewRootDepthOffset: number = 0,
  transientExpandedLeafNodeId: string | null = null,
  transientExpandedLeafMaxLines: number | null = null
): HierarchyPointNodeWithData | null => {
  if (!node) return null;
  const getChildrenFunction = overviewMode
    ? (n: VisualNode) => getVisibleChildrenForOverview(
      n,
      bigTokenEnabled,
      globalTopNNodes,
      topNRankByNodeId,
      centerTopNNodes
    )
    : (n: VisualNode) => getVisibleChildren(
      n,
      bigTokenEnabled,
      globalTopNNodes,
      topNRankByNodeId,
      centerTopNNodes
    );

  const isMergedPrefixUnderRootNode = (n: any): boolean => (
    n.depth === 1 && Boolean(n.data?.mergedNodes) && n.parent?.depth === 0
  );

  // 계층 구조 생성
  const hierarchy = d3.hierarchy(node, getChildrenFunction);
  const shouldForceTextExpanded = (nodeData: VisualNode): boolean => (
    transientExpandedLeafNodeId !== null && (
      nodeData.id === transientExpandedLeafNodeId
      || Boolean(nodeData.mergedNodes?.nodeIds?.includes(transientExpandedLeafNodeId))
    )
  );

  if (transientExpandedLeafNodeId !== null) {
    hierarchy.each(descendant => {
      if (!shouldForceTextExpanded(descendant.data)) return;
      descendant.data = {
        ...descendant.data,
        isTextExpanded: true,
        forcedTextMaxLines: transientExpandedLeafMaxLines ?? undefined
      };
    });
  }

  const pruneHierarchyForOverviewDepth = (
    hierarchyNode: d3.HierarchyNode<ExtendedVisualNode>,
    currentRelativeDepth: number
  ): void => {
    if (overviewMaxDepth === null) return;

    if (currentRelativeDepth >= overviewMaxDepth) {
      const hasBadDescendant = hierarchyNode
        .descendants()
        .slice(1)
        .some(descendant => (
          descendant.data.evaluationCategory === 'bad'
          || descendant.data.ancestorEvaluation === 'bad'
        ));

      hierarchyNode.children = undefined;
      hierarchyNode.data = {
        ...hierarchyNode.data,
        overviewDepthPrunedHasBadDescendant: hasBadDescendant || undefined,
        children: []
      };
      return;
    }

    hierarchyNode.children?.forEach(child => {
      pruneHierarchyForOverviewDepth(child, currentRelativeDepth + 1);
    });
  };

  if (overviewMode && overviewMaxDepth !== null) {
    pruneHierarchyForOverviewDepth(hierarchy, -overviewRootDepthOffset);
  }

  const root: HierarchyPointNodeWithData = (() => {
    if (useStandardTree) {
      const treeLayout = d3.tree<ExtendedVisualNode>()
        .nodeSize([40, 140]);

      return treeLayout(hierarchy) as HierarchyPointNodeWithData;
    } else {
      const treeLayout = leftAligned
        ? leftAlignedFlextree()
        : flextree();

      treeLayout
        .nodeSize((n: any): [number, number] => {
          // Overview 모드에서 필터링된 노드는 점 크기로 매우 작게
          if (overviewMode && n.data.isFiltered) {
            n.__layoutVerticalExtents = undefined;
            return [
              OVERVIEW_FILTERED_NODE_LAYOUT_HEIGHT,
              OVERVIEW_FILTERED_NODE_LAYOUT_WIDTH
            ] as [number, number];
          }

          // 비필터 노드는 overview 여부와 무관하게 동일한 실제 크기 계산
          const textLayout = calculateNodeTextLayout(n.data, undefined, {
            isRootNode: n.depth === 0,
            isMergedPrefixUnderRoot: isMergedPrefixUnderRootNode(n),
            isLeafNode: !n.children || n.children.length === 0
          });
          const collapsedTextLayout = calculateNodeTextLayout(n.data, NODE_MAX_LINES, {
            isRootNode: n.depth === 0,
            isMergedPrefixUnderRoot: isMergedPrefixUnderRootNode(n),
            isLeafNode: !n.children || n.children.length === 0
          });
          const hasChildren = n.data.children?.length > 0;
          const textWidth = textLayout.maxVisibleLineWidth;
          const totalWidth = textWidth + (hasChildren ? 40 : 0) + 24;
          const nodeHeight = textLayout.height + 10;

          if (!n.children || n.children.length === 0) {
            const collapsedNodeHeight = collapsedTextLayout.height + 10;
            n.__layoutVerticalExtents = {
              top: collapsedNodeHeight / 2,
              bottom: nodeHeight - (collapsedNodeHeight / 2)
            };
          } else {
            n.__layoutVerticalExtents = undefined;
          }

          return [nodeHeight, totalWidth] as [number, number];
        })
        .spacing(() => nodePadding);

      // 레이아웃 적용
      return treeLayout(hierarchy) as HierarchyPointNodeWithData;
    }
  })();

  if (centerTopNNodes && topNRankByNodeId) {
    centerTopNSubtreesWithinBands(
      root,
      topNRankByNodeId,
      overviewMode,
      nodePadding,
      isMergedPrefixUnderRootNode
    );
  }

  // 각 노드에 너비 정보 저장
  root.each(d => {
    // Overview 모드에서 필터링된 노드는 고정 너비
    if (overviewMode && d.data.isFiltered) {
      d.nodeWidth = OVERVIEW_FILTERED_NODE_DIAMETER;
    } else {
      const textLayout = calculateNodeTextLayout(d.data, undefined, {
        isRootNode: d.depth === 0,
        isMergedPrefixUnderRoot: isMergedPrefixUnderRootNode(d),
        isLeafNode: !d.children || d.children.length === 0
      });
      const hasChildren = d.data.children?.length > 0;
      d.nodeWidth = textLayout.maxVisibleLineWidth + (hasChildren ? 16 : 0) + 8;
    }
  });

  return root;
};

/**
 * 평가된 경로들의 총 누적 확률 계산
 */
export const calculateEvaluatedPathsProbability = (
  tree: VisualNode | null,
  evaluatedNodeIds: Set<string>
): number => {
  if (!tree || evaluatedNodeIds.size === 0) return 0;

  let totalProbability = 0;

  // 각 평가된 노드에 대해
  evaluatedNodeIds.forEach(nodeId => {
    const node = findNodeById(tree, nodeId);
    if (!node) return;

    // 루트부터 해당 노드까지의 확률 곱셈
    let pathProbability = 1.0;
    let currentNode: VisualNode | null = node;

    while (currentNode) {
      // 루트 노드가 아닐 때만 확률 곱하기
      if (currentNode.id !== tree.id) {
        pathProbability *= currentNode.prob;
      }
      // 부모 찾기
      currentNode = findParentNode(tree, currentNode.id);
    }

    totalProbability += pathProbability;
  });

  return totalProbability;
};

/**
 * 트리에서 노드의 부모를 찾는 헬퍼 함수
 */
const findParentNode = (tree: VisualNode, targetId: string): VisualNode | null => {
  // BFS로 부모 찾기
  const queue: VisualNode[] = [tree];

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (current.children) {
      for (const child of current.children) {
        if (child.id === targetId) {
          return current;
        }
        queue.push(child);
      }
    }
  }

  return null;
};

