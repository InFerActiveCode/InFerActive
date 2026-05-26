import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import { VisualNode, TokenSelection, CompletedSequence, EvaluatedNodes} from '../types/types';
import {
  TreeContainer,
  Tooltip,
  ZoomControl,
  HomeButton,
  VerticalSliderContainer,
  VerticalSlider,
  SliderButton,
  SliderText,
  SliderWithTicks,
  SliderTicks,
  SliderTick,
  ToggleControl,
  AlignmentButton,
  SankeyButton,
  MarkNotification,
  EvalActionBar,
  EvalActionButton
} from './tokenTreeStyles';
import {
  calculateNodeTextLayout,
  calculateTreeDimensions,
  getSelectedNodePath,
  createHierarchyData,
  createTopNSubtreeRoot,
  findPathToNode,
  findRightmostSinglePathNode,
  orderChildrenByTopNRank,
  passesEvaluationFilter,
  selectTopNLikeLeafNodes,
  HierarchyPointNodeWithData,
  ExtendedVisualNode,
  OVERVIEW_FILTERED_NODE_DIAMETER,
  OVERVIEW_FILTERED_NODE_RADIUS,
  OVERVIEW_NODE_PADDING
} from './tokenTreeUtils';
import {
  NodeContextMenu
} from './tokenTreeInteractions';
import { findNodeById, findParentNode } from '../utils/treeTransform';
import { renderVisualization } from './tokenTreeVisualization';

// TokenTreeVisualizer의 Props 정의
interface SystemSettings {
  leftAligned: boolean;
  useSankeyLinks: boolean;
  bigTokenEnabled: boolean;
  useStandardTree: boolean;
  overviewMode: boolean;
  toggleVisibility: 'all' | 'overview-only' | 'none';
  maxVisibleNodes: number;
  centerTopNNodes?: boolean;
}

export interface LeafPosition {
  x: number;
  y: number;
  nodeId: string;
}

interface TokenTreeVisualizerProps {
  node: ExtendedVisualNode;
  selectedToken: TokenSelection;
  completedSequences: CompletedSequence[];
  tempSequence?: CompletedSequence | null;
  onNodePin?: (nodeId: string) => void;
  onNodeEvaluate?: (nodeId: string, category: 'good' | 'bad') => void;
  onGenerateFromNode?: (nodeId: string) => Promise<any>;
  isNodeGenerating?: (nodeId: string) => boolean;
  onTokenSelect?: (sequenceId: string, tokenIndex: number) => void;
  onLeafPositionChange?: (pos: LeafPosition | null) => void;
  evaluatedNodeIds?: EvaluatedNodes;
  evaluatedPathsTotal?: number;
  evaluationFilters?: {
    showGood: boolean;
    showBad: boolean;
    showUnmarked: boolean;
  };
  evaluationColorEnabled?: boolean;
  systemSettings?: SystemSettings;
  externalOverviewEnabled?: boolean;
  externalOverviewMaxDepth?: number;
  filteringMinProb?: number;
  filteringMaxBranches?: number;
  showLayoutToggles?: boolean;
}

const ROOT_VIEW_MARGIN = { top: 70, right: 120, bottom: 20, left: 90 };

const MIN_VISIBLE_NODES = 1;
const MAX_VISIBLE_NODES = 50;
const DEFAULT_OVERVIEW_RENDER_MAX_DEPTH = 3;
const MERGE_RELEASE_TRANSITION_MS = 180;
const MERGE_RELEASE_HINT_SUPPRESS_MS = 500;
const DIRECT_LEFT_FOLD_DELAY_MS = 300;
const clampVisibleNodes = (value: number): number => {
  if (Number.isNaN(value)) return MIN_VISIBLE_NODES;
  return Math.min(Math.max(value, MIN_VISIBLE_NODES), MAX_VISIBLE_NODES);
};

export const TokenTreeVisualizer: React.FC<TokenTreeVisualizerProps> = ({
  node,
  selectedToken,
  completedSequences,
  tempSequence,
  onNodePin,
  onNodeEvaluate,
  onGenerateFromNode,
  isNodeGenerating,
  onTokenSelect,
  onLeafPositionChange,
  evaluatedNodeIds,
  evaluatedPathsTotal,
  evaluationFilters,
  evaluationColorEnabled = true,
  systemSettings,
  externalOverviewEnabled = false,
  externalOverviewMaxDepth,
  filteringMinProb = 0,
  filteringMaxBranches = MAX_VISIBLE_NODES,
  showLayoutToggles = true,
}) => {
  const showCenterTopNToggle = false;

  // DOM 참조
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // D3 변환 및 줌 참조
  const transformRef = useRef<d3.ZoomTransform | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const previousRootRef = useRef<VisualNode | null>(null);
  const latestTreeRef = useRef<ExtendedVisualNode | null>(node);

  const explicitResetRef = useRef(false);

  const initializedRef = useRef(false);
  const mergeReleaseTransitionUntilRef = useRef(0);
  const mergeReleaseHintSuppressUntilRef = useRef(0);
  const mergeReleaseTransitionTimerRef = useRef<number | null>(null);
  const mergeReleaseHintSuppressTimerRef = useRef<number | null>(null);
  const mergeReleaseExpandTimerRef = useRef<number | null>(null);
  const mergeReleaseExpandedNodeRef = useRef<{
    nodeId: string;
    previousIsExpanded: boolean | undefined;
  } | null>(null);
  const lastNodeClickTargetRef = useRef<string | null>(null);
  const directLeftFoldTimerRef = useRef<Map<string, number>>(new Map());
  const directLeftFoldHiddenIdsRef = useRef<Map<string, Set<string>>>(new Map());

  const [maxVisibleNodes, setMaxVisibleNodes] = useState(
    clampVisibleNodes(systemSettings?.maxVisibleNodes ?? 5)
  );
  const discreteNodeCounts = [5, 10, 20, 30, 40, 50];
  const [treeData, setTreeData] = useState<HierarchyPointNodeWithData | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [treeUpdateTrigger, setTreeUpdateTrigger] = useState(0);
  const [manuallyExpandedLeafNodeId, setManuallyExpandedLeafNodeId] = useState<string | null>(null);
  const [leftAligned, setLeftAligned] = useState(systemSettings?.leftAligned ?? true);
  const [useSankeyLinks, setUseSankeyLinks] = useState(systemSettings?.useSankeyLinks ?? true);
  const [bigTokenEnabled, setBigTokenEnabled] = useState(systemSettings?.bigTokenEnabled ?? true);
  const [useStandardTree, setUseStandardTree] = useState(systemSettings?.useStandardTree ?? false);
  const [overviewMode, setOverviewMode] = useState(systemSettings?.overviewMode ?? false);
  const [centerTopNNodes, setCenterTopNNodes] = useState(systemSettings?.centerTopNNodes ?? false);
  const [topNRootNodeId, setTopNRootNodeId] = useState<string>('root');
  const [rerootPreviewVisibleIds, setRerootPreviewVisibleIds] = useState<Set<string> | null>(null);
  const [rerootHoverPreviewTargetNodeId, setRerootHoverPreviewTargetNodeId] = useState<string | null>(null);
  const [transientHiddenNodeIds, setTransientHiddenNodeIds] = useState<Set<string>>(new Set());
  const rerootPreviewTimerRef = useRef<number | null>(null);
  const previousRootChildrenSignatureRef = useRef<string | null>(null);

  const treeSnapshot = useMemo(() => ({
    root: node,
    revision: treeUpdateTrigger
  }), [node, treeUpdateTrigger]);

  const getUserVisibleChildren = useCallback((targetNode: VisualNode): VisualNode[] => {
    return targetNode.children.filter(child => child.isUserFolded !== true);
  }, []);

  const effectiveVisibleRootId = useMemo(() => {
    const rootNode = treeSnapshot.root;
    if (!rootNode) return 'root';

    let current = topNRootNodeId === 'root'
      ? rootNode
      : (findNodeById(rootNode, topNRootNodeId) ?? rootNode);

    while (true) {
      const visibleChildren = getUserVisibleChildren(current);
      if (visibleChildren.length !== 1) break;
      current = visibleChildren[0];
    }

    return current.id;
  }, [treeSnapshot, topNRootNodeId, getUserVisibleChildren]);

  const effectiveVisibleRootPathNodeIds = useMemo(() => {
    if (!node || effectiveVisibleRootId === 'root') return [];

    const globalRootId = node.id;
    return findPathToNode(node, effectiveVisibleRootId)
      .map(pathNode => pathNode.id)
      .filter(pathNodeId => pathNodeId !== globalRootId);
  }, [node, effectiveVisibleRootId]);

  const isEffectiveVisibleRootPrefixNode = useCallback((nodeRef: any): boolean => {
    if (effectiveVisibleRootId === 'root') return false;
    if (nodeRef?.depth !== 1 || nodeRef?.parent?.depth !== 0) return false;

    const nodeData = nodeRef?.data ?? nodeRef;
    if (!nodeData) return false;

    const representedNodeIds = nodeData.mergedNodes?.nodeIds && nodeData.mergedNodes.nodeIds.length > 0
      ? nodeData.mergedNodes.nodeIds
      : (nodeData.id ? [nodeData.id] : []);

    return representedNodeIds.includes(effectiveVisibleRootId);
  }, [effectiveVisibleRootId]);

  const effectiveOverviewMode = overviewMode || externalOverviewEnabled;
  const defaultNodePadding = 20;
  const nodePadding = effectiveOverviewMode ? OVERVIEW_NODE_PADDING : defaultNodePadding;
  const overviewRenderMaxDepth = Math.max(
    1,
    Math.floor(externalOverviewMaxDepth ?? DEFAULT_OVERVIEW_RENDER_MAX_DEPTH)
  );

  const syncDirectLeftFoldHiddenNodeIds = useCallback(() => {
    const nextHiddenNodeIds = new Set<string>();
    directLeftFoldHiddenIdsRef.current.forEach(hiddenIds => {
      hiddenIds.forEach(hiddenNodeId => {
        nextHiddenNodeIds.add(hiddenNodeId);
      });
    });
    setTransientHiddenNodeIds(nextHiddenNodeIds);
  }, []);

  const triggerMergeReleaseTransition = useCallback(() => {
    mergeReleaseTransitionUntilRef.current = Date.now() + MERGE_RELEASE_TRANSITION_MS;
    mergeReleaseHintSuppressUntilRef.current = Date.now() + MERGE_RELEASE_HINT_SUPPRESS_MS;
    if (mergeReleaseTransitionTimerRef.current !== null) {
      window.clearTimeout(mergeReleaseTransitionTimerRef.current);
    }
    if (mergeReleaseHintSuppressTimerRef.current !== null) {
      window.clearTimeout(mergeReleaseHintSuppressTimerRef.current);
    }
    mergeReleaseTransitionTimerRef.current = window.setTimeout(() => {
      mergeReleaseTransitionUntilRef.current = 0;
      mergeReleaseTransitionTimerRef.current = null;
    }, MERGE_RELEASE_TRANSITION_MS + 32);
    mergeReleaseHintSuppressTimerRef.current = window.setTimeout(() => {
      mergeReleaseHintSuppressUntilRef.current = 0;
      mergeReleaseHintSuppressTimerRef.current = null;
    }, MERGE_RELEASE_HINT_SUPPRESS_MS + 32);
  }, []);

  useEffect(() => {
    latestTreeRef.current = node;
  }, [node]);

  const restoreTemporaryMergeReleaseExpansion = useCallback((): boolean => {
    const pendingExpansion = mergeReleaseExpandedNodeRef.current;
    mergeReleaseExpandedNodeRef.current = null;

    if (!pendingExpansion) return false;

    const activeTree = latestTreeRef.current;
    if (!activeTree) return false;

    const expandedNode = findNodeById(activeTree, pendingExpansion.nodeId);
    if (!expandedNode) return false;

    expandedNode.isExpanded = pendingExpansion.previousIsExpanded;
    return true;
  }, []);

  const resetPendingMergeReleaseExpansion = useCallback((): boolean => {
    if (mergeReleaseExpandTimerRef.current !== null) {
      window.clearTimeout(mergeReleaseExpandTimerRef.current);
      mergeReleaseExpandTimerRef.current = null;
    }

    return restoreTemporaryMergeReleaseExpansion();
  }, [restoreTemporaryMergeReleaseExpansion]);

  // 사용자 폴딩 상태 초기화 유틸리티
  const resetUserFolding = useCallback((targetNode: VisualNode) => {
    targetNode.isUserFolded = undefined;
    targetNode.isExpanded = undefined;
    targetNode.children.forEach(resetUserFolding);
  }, []);

  useEffect(() => {
    const directLeftFoldTimers = directLeftFoldTimerRef.current;
    const directLeftFoldHiddenIds = directLeftFoldHiddenIdsRef.current;

    return () => {
      if (mergeReleaseTransitionTimerRef.current !== null) {
        window.clearTimeout(mergeReleaseTransitionTimerRef.current);
      }
      if (mergeReleaseHintSuppressTimerRef.current !== null) {
        window.clearTimeout(mergeReleaseHintSuppressTimerRef.current);
      }
      if (mergeReleaseExpandTimerRef.current !== null) {
        window.clearTimeout(mergeReleaseExpandTimerRef.current);
      }
      restoreTemporaryMergeReleaseExpansion();
      if (rerootPreviewTimerRef.current !== null) {
        window.clearTimeout(rerootPreviewTimerRef.current);
      }
      directLeftFoldTimers.forEach(timerId => {
        window.clearTimeout(timerId);
      });
      directLeftFoldTimers.clear();
      directLeftFoldHiddenIds.clear();
    };
  }, [restoreTemporaryMergeReleaseExpansion]);

  // 필터링 전 뷰포트 중심점 기억
  const viewportCenterRef = useRef<{ x: number; y: number } | null>(null);

  // 시스템 설정이 변경될 때 토글 상태 업데이트
  useEffect(() => {
    if (systemSettings) {
      setLeftAligned(systemSettings.leftAligned);
      setUseSankeyLinks(systemSettings.useSankeyLinks);
      setBigTokenEnabled(systemSettings.bigTokenEnabled);
      setUseStandardTree(systemSettings.useStandardTree);
      setOverviewMode(systemSettings.overviewMode);
      setCenterTopNNodes(systemSettings.centerTopNNodes ?? false);
      setMaxVisibleNodes(clampVisibleNodes(systemSettings.maxVisibleNodes));
    }
  }, [systemSettings]);

  useEffect(() => {
    explicitResetRef.current = true;
    viewportCenterRef.current = null;
  }, [effectiveOverviewMode]);

  // Top-N 서브루트가 현재 트리에 없으면 전역 루트로 복귀
  useEffect(() => {
    if (!node || topNRootNodeId === 'root') return;
    const rootNode = findNodeById(node, topNRootNodeId);
    if (!rootNode) {
      setTopNRootNodeId('root');
      return;
    }

    const rightmostNode = findRightmostSinglePathNode(node, topNRootNodeId);
    if (rightmostNode && rightmostNode.id !== topNRootNodeId) {
      setTopNRootNodeId(rightmostNode.id);
    }
  }, [node, topNRootNodeId, treeUpdateTrigger]);

  // 컨텍스트 메뉴 상태
  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    position: { x: number, y: number };
    nodeData: any;
  }>({
    visible: false,
    position: { x: 0, y: 0 },
    nodeData: null
  });

  // 평가 액션바 상태
  const [evalActionBar, setEvalActionBar] = useState<{
    visible: boolean;
    x: number;
    y: number;
    nodeId: string;
    nodeData: any;
  } | null>(null);

  // 루트 재설정 팝업 상태 (prefix merged node 토큰 클릭 시)
  const [rootPopup, setRootPopup] = useState<{
    visible: boolean;
    x: number;
    y: number;
    targetNodeId: string;
  } | null>(null);

  // 마크 알림 상태
  const [markNotifications, setMarkNotifications] = useState<Array<{
    id: string;
    x: number;
    y: number;
    category: 'good' | 'bad';
    timestamp: number;
  }>>([]);


  const handleNodeEvaluate = useCallback((nodeId: string, category: 'good' | 'bad') => {
    if (!onNodeEvaluate) return;

    onNodeEvaluate(nodeId, category);

    if (svgRef.current && containerRef.current) {
      const nodeElement = d3.select(svgRef.current)
        .selectAll(`[data-node-id="${nodeId}"]`)
        .node() as SVGGElement;

      if (nodeElement) {
        const bbox = nodeElement.getBoundingClientRect();
        const containerRect = containerRef.current.getBoundingClientRect();

        const relativeX = bbox.left - containerRect.left + bbox.width / 2;
        const relativeY = bbox.top - containerRect.top;

        const notification = {
          id: `${nodeId}-${Date.now()}`,
          x: relativeX,
          y: relativeY,
          category,
          timestamp: Date.now()
        };

        setMarkNotifications(prev => [...prev, notification]);

        setTimeout(() => {
          setMarkNotifications(prev => prev.filter(n => n.id !== notification.id));
        }, 1000);
      }
    }
  }, [onNodeEvaluate]);

  const handleGenerateFromNode = useCallback(async (nodeId: string) => {
    if (!onGenerateFromNode) {
      return null;
    }

    try {
      return await onGenerateFromNode(nodeId);
    } catch {
      return null;
    }
  }, [onGenerateFromNode]);

  const selectedNodePath = useMemo(() => {
    const ensureRootInPath = (path: string[]): string[] => {
      if (!node || path.length === 0) return path;
      return path[0] === node.id ? path : [node.id, ...path];
    };

    if (tempSequence && selectedToken.sequenceId === tempSequence.id) {
      const selectedTokenInfo = tempSequence.tokens[selectedToken.tokenIndex || 0];
      if (!selectedTokenInfo) return [];

      const tempPath = tempSequence.tokens
        .slice(0, (selectedToken.tokenIndex || 0) + 1)
        .map(t => t.nodeId);
      return ensureRootInPath(tempPath);
    }

    const completedPath = getSelectedNodePath(completedSequences, selectedToken);
    if (completedPath.length > 0) {
      return ensureRootInPath(completedPath);
    }

    if (node && selectedToken.sequenceId?.startsWith('temp-')) {
      const tempNodeId = selectedToken.sequenceId.replace('temp-', '');
      if (tempNodeId) {
        const pathToTempNode = findPathToNode(node, tempNodeId);
        if (pathToTempNode.length > 0) {
          return pathToTempNode.map(pathNode => pathNode.id);
        }
      }
    }

    return [];
  }, [selectedToken, completedSequences, tempSequence, node]);

  // 트리 접기/펴기 핸들러
  const toggleFold = useCallback((nodeId: string) => {
    // 원본 데이터(node)에서 해당 노드 찾기
    if (!node) return;

    const parentNode = findNodeById(node, nodeId);
    if (!parentNode || !parentNode.children || parentNode.children.length === 0) return;

    // 모든 자식이 펴져있는지 확인 (isFolded가 false이거나 undefined인 경우)
    const allUnfolded = parentNode.children.every(child =>
      child.isFolded === false || child.isFolded === undefined
    );

    // 모든 자식이 펴져있으면 모두 접기, 하나라도 접혀있으면 모두 펼치기
    const newFoldState = allUnfolded;

    parentNode.children.forEach(child => {
      child.isFolded = newFoldState;
      child.isUserFolded = newFoldState;
    });

    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  const effectiveTopNRoot = useMemo(() => {
    const rootNode = treeSnapshot.root;
    if (!rootNode) return { node: null, revision: treeSnapshot.revision };
    if (effectiveVisibleRootId === 'root') {
      return { node: rootNode, revision: treeSnapshot.revision };
    }
    return {
      node: createTopNSubtreeRoot(rootNode, effectiveVisibleRootId) || rootNode,
      revision: treeSnapshot.revision
    };
  }, [treeSnapshot, effectiveVisibleRootId]);

  // Top-N(리프 개수 기준): frontier(BFS) 순서를 유지하며 visible leaf 수를 target에 맞춤
  const topNSelection = useMemo(() => {
    const topNRoot = effectiveTopNRoot.node;
    if (!topNRoot) return undefined;

    if (maxVisibleNodes === 0) {
      return {
        nodeIds: new Set<string>(),
        rankByNodeId: new Map<string, number>()
      };
    }

    const topNodes = new Set<string>();
    const rankByNodeId = new Map<string, number>();
    const target = maxVisibleNodes;
    const minCumulativeProb = Math.max(0, filteringMinProb) / 100;
    const maxBranches = Math.max(1, Math.floor(filteringMaxBranches));

    const isTopNEligible = (candidate: VisualNode): boolean => (
      // Keep the original Top-N selection stable even after manual folding.
      // User folding is applied later during rendering, so folded nodes simply
      // disappear instead of being replaced by the next candidates.
      passesEvaluationFilter(candidate, evaluationFilters)
    );

    const getCumulativeProb = (candidate: VisualNode): number => (
      candidate.cumulativeProb ?? candidate.prob ?? 0
    );

    const getGreedyChild = (children: VisualNode[]): VisualNode | null => {
      if (children.length === 0) return null;
      return children.reduce((best, child) => (
        getCumulativeProb(child) > getCumulativeProb(best) ? child : best
      ));
    };

    const getValidChildren = (n: VisualNode): VisualNode[] => {
      // Top-N folding should follow the same sibling order the tree renders with,
      // otherwise lower nodes can survive while higher siblings are folded.
      const eligibleChildren = n.children.filter(isTopNEligible);
      const greedyChild = getGreedyChild(eligibleChildren);
      const minProbFilteredChildren = eligibleChildren.filter(child => (
        child.id === greedyChild?.id
        || minCumulativeProb <= 0
        || getCumulativeProb(child) >= minCumulativeProb
      ));
      const cappedChildIds = new Set(
        minProbFilteredChildren.slice(0, maxBranches).map(child => child.id)
      );

      return eligibleChildren.filter(child => (
        cappedChildIds.has(child.id) || child.id === greedyChild?.id
      ));
    };

    const selectedLeafNodes = selectTopNLikeLeafNodes(
      topNRoot,
      target,
      getValidChildren
    );

    const greedyParentQueue: VisualNode[] = [topNRoot];
    const queuedGreedyParentIds = new Set<string>([topNRoot.id]);
    const enqueueGreedyParent = (candidate: VisualNode) => {
      if (queuedGreedyParentIds.has(candidate.id)) return;
      queuedGreedyParentIds.add(candidate.id);
      greedyParentQueue.push(candidate);
    };

    selectedLeafNodes.forEach((leaf, rank) => {
      const path = findPathToNode(topNRoot, leaf.id);
      for (const pathNode of path.slice(1)) {
        topNodes.add(pathNode.id);
        enqueueGreedyParent(pathNode);
        const existingRank = rankByNodeId.get(pathNode.id);
        if (existingRank === undefined || rank < existingRank) {
          rankByNodeId.set(pathNode.id, rank);
        }
      }
    });

    for (let index = 0; index < greedyParentQueue.length; index += 1) {
      const parent = greedyParentQueue[index];
      const greedyChild = getGreedyChild(parent.children.filter(isTopNEligible));
      if (!greedyChild) continue;

      topNodes.add(greedyChild.id);
      if (!rankByNodeId.has(greedyChild.id)) {
        rankByNodeId.set(greedyChild.id, rankByNodeId.get(parent.id) ?? 0);
      }
      enqueueGreedyParent(greedyChild);
    }

    return {
      nodeIds: topNodes,
      rankByNodeId
    };
  }, [effectiveTopNRoot, maxVisibleNodes, evaluationFilters, filteringMinProb, filteringMaxBranches]);

  const effectiveTopNNodes = topNSelection?.nodeIds;
  const topNRankByNodeId = topNSelection?.rankByNodeId;
  const hierarchyRootNode = effectiveTopNRoot.node;
  const overviewDepthOffset = effectiveVisibleRootId === 'root' ? 0 : 1;

  const baseHierarchyData = useMemo(() => {
    return createHierarchyData(
      hierarchyRootNode,
      nodePadding,
      leftAligned,
      bigTokenEnabled,
      evaluationFilters,
      effectiveTopNNodes,
      useStandardTree,
      effectiveOverviewMode,
      centerTopNNodes,
      topNRankByNodeId,
      effectiveOverviewMode ? overviewRenderMaxDepth : null,
      overviewDepthOffset,
      null,
      null
    );
  }, [
    hierarchyRootNode,
    nodePadding,
    leftAligned,
    bigTokenEnabled,
    evaluationFilters,
    effectiveTopNNodes,
    useStandardTree,
    effectiveOverviewMode,
    centerTopNNodes,
    topNRankByNodeId,
    overviewRenderMaxDepth,
    overviewDepthOffset
  ]);

  const selectedAutoExpandedLeafNodeId = useMemo(() => {
    const selectedTerminalNodeId = selectedNodePath.length > 0
      ? selectedNodePath[selectedNodePath.length - 1]
      : null;
    if (!baseHierarchyData || !selectedTerminalNodeId) return null;

    const renderedNode = baseHierarchyData.descendants().find(desc => {
      if (desc.data.id === selectedTerminalNodeId) return true;
      return Boolean(desc.data.mergedNodes?.nodeIds?.includes(selectedTerminalNodeId));
    });
    if (!renderedNode || (renderedNode.children && renderedNode.children.length > 0)) {
      return null;
    }

    const collapsedLayout = calculateNodeTextLayout(renderedNode.data, 1, {
      isRootNode: renderedNode.depth === 0,
      isMergedPrefixUnderRoot: isEffectiveVisibleRootPrefixNode(renderedNode),
      isLeafNode: true
    });

    return collapsedLayout.truncated ? selectedTerminalNodeId : null;
  }, [baseHierarchyData, selectedNodePath, isEffectiveVisibleRootPrefixNode]);

  useEffect(() => {
    if (manuallyExpandedLeafNodeId === null) return;
    if (manuallyExpandedLeafNodeId === selectedAutoExpandedLeafNodeId) return;
    setManuallyExpandedLeafNodeId(null);
  }, [manuallyExpandedLeafNodeId, selectedAutoExpandedLeafNodeId]);

  const selectedExpandedLeafMaxLines = manuallyExpandedLeafNodeId === selectedAutoExpandedLeafNodeId
    ? 6
    : null;

  const handleExpandSelectedLeafText = useCallback((nodeId: string) => {
    setManuallyExpandedLeafNodeId(previousNodeId => (
      previousNodeId === nodeId ? null : nodeId
    ));
  }, []);

  // 트리 계층 데이터 계산
  const hierarchyData = useMemo(() => {
    return createHierarchyData(
      hierarchyRootNode,
      nodePadding,
      leftAligned,
      bigTokenEnabled,
      evaluationFilters,
      effectiveTopNNodes,
      useStandardTree,
      effectiveOverviewMode,
      centerTopNNodes,
      topNRankByNodeId,
      effectiveOverviewMode ? overviewRenderMaxDepth : null,
      overviewDepthOffset,
      selectedAutoExpandedLeafNodeId,
      selectedExpandedLeafMaxLines
    );
  }, [
    hierarchyRootNode,
    nodePadding,
    leftAligned,
    bigTokenEnabled,
    evaluationFilters,
    effectiveTopNNodes,
    useStandardTree,
    effectiveOverviewMode,
    centerTopNNodes,
    topNRankByNodeId,
    overviewRenderMaxDepth,
    overviewDepthOffset,
    selectedAutoExpandedLeafNodeId,
    selectedExpandedLeafMaxLines
  ]);

  const handleResetToGlobalRoot = useCallback(() => {
    if (rerootPreviewTimerRef.current !== null) {
      window.clearTimeout(rerootPreviewTimerRef.current);
      rerootPreviewTimerRef.current = null;
    }
    setRerootPreviewVisibleIds(null);
    explicitResetRef.current = true;
    viewportCenterRef.current = null;

    directLeftFoldTimerRef.current.forEach(timerId => {
      window.clearTimeout(timerId);
    });
    directLeftFoldTimerRef.current.clear();
    directLeftFoldHiddenIdsRef.current.clear();
    syncDirectLeftFoldHiddenNodeIds();

    resetPendingMergeReleaseExpansion();

    if (node) {
      resetUserFolding(node);
    }

    setTopNRootNodeId('root');
    if (onTokenSelect) {
      onTokenSelect('', -1);
    }
    setEvalActionBar(null);
    setRootPopup(null);
    setContextMenu(prev => ({ ...prev, visible: false }));
    lastNodeClickTargetRef.current = null;
    setTreeUpdateTrigger(prev => prev + 1);
  }, [node, onTokenSelect, resetPendingMergeReleaseExpansion, resetUserFolding, syncDirectLeftFoldHiddenNodeIds]);

  useEffect(() => {
    const rootChildrenSignature = node
      ? node.children.map(child => child.id).join('|')
      : null;
    const previousRootChildrenSignature = previousRootChildrenSignatureRef.current;

    previousRootChildrenSignatureRef.current = rootChildrenSignature;

    if (
      topNRootNodeId !== 'root' &&
      previousRootChildrenSignature !== null &&
      rootChildrenSignature !== previousRootChildrenSignature
    ) {
      handleResetToGlobalRoot();
    }
  }, [node, topNRootNodeId, handleResetToGlobalRoot]);


  const selectNodeFromVisualNode = useCallback((d: HierarchyPointNodeWithData, preferredNodeId?: string) => {
    if (!onTokenSelect) return;

    let selectedSequence = null;
    let selectedTokenIndex = -1;

    const targetNodeId = preferredNodeId ?? (d.data.mergedNodes?.nodeIds
      ? d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1]
      : d.data.id);

    if (!targetNodeId) return;

    for (const seq of completedSequences) {
      const tokenIdx = seq.tokens.findIndex(t => t.nodeId === targetNodeId);
      if (tokenIdx >= 0) {
        selectedSequence = seq;
        selectedTokenIndex = tokenIdx;
        break;
      }
    }

    if (selectedSequence && selectedTokenIndex >= 0) {
      onTokenSelect(selectedSequence.id, selectedTokenIndex);
    } else {
      const tempSequenceId = `temp-${targetNodeId}`;
      onTokenSelect(tempSequenceId, 0);
    }
  }, [onTokenSelect, completedSequences]);

  const isRerootBlockedNode = useCallback((targetNodeId: string): boolean => {
    if (!node) return false;
    const targetNode = findNodeById(node, targetNodeId);
    if (!targetNode) return false;
    if (targetNode.id === node.id) return false;

    // "실제 리프로 끝나는 재귀적 단일 경로" 구간만 reroot 차단
    const isRecursiveSinglePathToLeaf = (current: VisualNode): boolean => {
      if (!current.children || current.children.length === 0) {
        return true;
      }
      if (current.children.length !== 1) {
        return false;
      }
      return isRecursiveSinglePathToLeaf(current.children[0]);
    };

    return isRecursiveSinglePathToLeaf(targetNode);
  }, [node]);

  const collectSubtreeNodeIds = useCallback((subtreeRoot: VisualNode): Set<string> => {
    const ids = new Set<string>();
    const stack: VisualNode[] = [subtreeRoot];

    while (stack.length > 0) {
      const current = stack.pop();
      if (!current) continue;
      ids.add(current.id);
      for (const child of current.children) {
        stack.push(child);
      }
    }

    return ids;
  }, []);

  const collectActualSingleBranchNodeIds = useCallback((startNodeId: string): string[] => {
    if (!node) return [];

    const centerNode = findNodeById(node, startNodeId);
    if (!centerNode) return [];

    const branchNodeIds: string[] = [centerNode.id];

    let upwardNode = centerNode;
    while (true) {
      const parentNode = findParentNode(node, upwardNode.id);
      if (!parentNode || parentNode.children.length !== 1) break;
      branchNodeIds.unshift(parentNode.id);
      upwardNode = parentNode;
    }

    let downwardNode = centerNode;
    while (downwardNode.children.length === 1) {
      const nextNode = downwardNode.children[0];
      if (!nextNode || isRerootBlockedNode(nextNode.id)) break;
      downwardNode = nextNode;
      branchNodeIds.push(downwardNode.id);
    }

    return branchNodeIds;
  }, [node, isRerootBlockedNode]);

  const applyRerootWithPreview = useCallback((clickedNodeId: string, nextRootNodeId: string) => {
    if (!node) return;

    if (rerootPreviewTimerRef.current !== null) {
      window.clearTimeout(rerootPreviewTimerRef.current);
      rerootPreviewTimerRef.current = null;
    }

    // Preview 단계는 잠시 비활성화하고 즉시 reroot만 수행한다.
    void clickedNodeId;
    setRerootPreviewVisibleIds(null);

    if (nextRootNodeId === topNRootNodeId) {
      return;
    }

    setTopNRootNodeId(nextRootNodeId);
  }, [node, topNRootNodeId]);

  const getVisibleAnchorBounds = useCallback((element: Element): { left: number; right: number; top: number; bottom: number } | null => {
    const isSvgTextElement = typeof SVGTextContentElement !== 'undefined' && element instanceof SVGTextContentElement;
    if (!isSvgTextElement || !svgRef.current) {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 || rect.height > 0
        ? { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }
        : null;
    }

    const textElement = element as SVGTextContentElement;
    const text = textElement.textContent ?? '';
    const charCount = typeof textElement.getNumberOfChars === 'function'
      ? textElement.getNumberOfChars()
      : 0;
    const screenCTM = typeof textElement.getScreenCTM === 'function'
      ? textElement.getScreenCTM()
      : null;

    if (!screenCTM || charCount <= 0) {
      const rect = textElement.getBoundingClientRect();
      return rect.width > 0 || rect.height > 0
        ? { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }
        : null;
    }

    const point = svgRef.current.createSVGPoint();
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let hasVisibleChar = false;

    for (let i = 0; i < charCount; i++) {
      const char = text[i] ?? '';
      if (char.trim().length === 0) continue;

      let charRect: DOMRect;
      try {
        charRect = textElement.getExtentOfChar(i);
      } catch {
        continue;
      }

      const corners: Array<[number, number]> = [
        [charRect.x, charRect.y],
        [charRect.x + charRect.width, charRect.y],
        [charRect.x, charRect.y + charRect.height],
        [charRect.x + charRect.width, charRect.y + charRect.height]
      ];

      for (const [x, y] of corners) {
        point.x = x;
        point.y = y;
        const screenPoint = point.matrixTransform(screenCTM);
        minX = Math.min(minX, screenPoint.x);
        maxX = Math.max(maxX, screenPoint.x);
        minY = Math.min(minY, screenPoint.y);
        maxY = Math.max(maxY, screenPoint.y);
      }

      hasVisibleChar = true;
    }

    if (!hasVisibleChar) {
      const rect = textElement.getBoundingClientRect();
      return rect.width > 0 || rect.height > 0
        ? { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }
        : null;
    }

    return { left: minX, right: maxX, top: minY, bottom: maxY };
  }, []);

  const setRootPopupFromAnchors = useCallback((
    targetNodeId: string,
    anchorElements?: ArrayLike<Element> | null,
    fallbackAnchorElement?: Element | null
  ) => {
    if (!containerRef.current) return;

    const resolvedAnchorElements = Array.from(anchorElements ?? []).filter((element): element is Element => Boolean(element));
    const candidateElements = resolvedAnchorElements.length > 0
      ? resolvedAnchorElements
      : (fallbackAnchorElement ? [fallbackAnchorElement] : []);
    if (candidateElements.length === 0) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const anchorRects = candidateElements
      .map(getVisibleAnchorBounds)
      .filter((rect): rect is { left: number; right: number; top: number; bottom: number } => rect !== null);
    if (anchorRects.length === 0) return;

    const left = Math.min(...anchorRects.map(rect => rect.left));
    const right = Math.max(...anchorRects.map(rect => rect.right));
    const top = Math.min(...anchorRects.map(rect => rect.top));

    setRootPopup({
      visible: true,
      x: left - containerRect.left + ((right - left) / 2),
      y: top - containerRect.top,
      targetNodeId
    });
  }, [getVisibleAnchorBounds]);

  const handleRerootButtonAnchorHover = useCallback((
    targetNodeId: string,
    anchorElements?: ArrayLike<Element> | null,
    fallbackAnchorElement?: Element | null
  ) => {
    setRootPopupFromAnchors(targetNodeId, anchorElements, fallbackAnchorElement ?? null);
  }, [setRootPopupFromAnchors]);

  const handleNodeClick = useCallback((event: React.MouseEvent | MouseEvent, d: HierarchyPointNodeWithData) => {
    if (event.button !== 0) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    if (d.depth === 0 && d.data.id === 'root') {
      handleResetToGlobalRoot();
      return;
    }

    if (!node) return;

    const clickableNodeIds = d.data.mergedNodes?.nodeIds && d.data.mergedNodes.nodeIds.length > 0
      ? d.data.mergedNodes.nodeIds
      : [d.data.id];
    const selectedTerminalNodeId = selectedNodePath.length > 0
      ? selectedNodePath[selectedNodePath.length - 1]
      : null;
    const isDirectlySelected = selectedTerminalNodeId !== null
      && clickableNodeIds.includes(selectedTerminalNodeId);
    const rerootTargetNodeId = clickableNodeIds[clickableNodeIds.length - 1];
    const isRepeatedNodeClick = rerootTargetNodeId !== undefined
      && rerootTargetNodeId !== null
      && lastNodeClickTargetRef.current === rerootTargetNodeId;
    // 이미 선택된 노드를 다시 클릭하면 선택 해제
    if (isDirectlySelected || isRepeatedNodeClick) {
      if (onTokenSelect) onTokenSelect('', -1);
      lastNodeClickTargetRef.current = null;
      setEvalActionBar(null);
      setRootPopup(null);
      setContextMenu(prev => ({ ...prev, visible: false }));
      return;
    }

    if (rerootPreviewTimerRef.current !== null) {
      window.clearTimeout(rerootPreviewTimerRef.current);
      rerootPreviewTimerRef.current = null;
    }
    setRerootPreviewVisibleIds(null);

    // 새 노드 선택
    selectNodeFromVisualNode(d, rerootTargetNodeId);
    lastNodeClickTargetRef.current = rerootTargetNodeId ?? null;
    setEvalActionBar(null);
    setRootPopup(null);

    setContextMenu(prev => ({ ...prev, visible: false }));
  }, [node, selectedNodePath, selectNodeFromVisualNode, onTokenSelect, handleResetToGlobalRoot]);

  // 토큰 클릭 핸들러: reroot 대신 선택 상태와 root 버튼 위치만 갱신
  const handleMergedRootTokenClick = useCallback((event: React.MouseEvent | MouseEvent, d: HierarchyPointNodeWithData, tokenIndex: number) => {
    if (event.button !== 0) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    if (!node) return;

    const clickableNodeIds = d.data.mergedNodes?.nodeIds && d.data.mergedNodes.nodeIds.length > 0
      ? d.data.mergedNodes.nodeIds
      : [d.data.id];
    const clickedNodeId = clickableNodeIds[tokenIndex] ?? clickableNodeIds[clickableNodeIds.length - 1];
    if (!clickedNodeId) return;

    // 토큰 클릭은 현재 terminal selection과 정확히 일치할 때만 해제로 본다.
    const selectedTerminalNodeId = selectedNodePath.length > 0
      ? selectedNodePath[selectedNodePath.length - 1]
      : null;
    const isCurrentlySelected = selectedTerminalNodeId !== null
      && clickableNodeIds.includes(selectedTerminalNodeId);

    if (isCurrentlySelected) {
      if (onTokenSelect) onTokenSelect('', -1);
      lastNodeClickTargetRef.current = null;
      setEvalActionBar(null);
      setContextMenu(prev => ({ ...prev, visible: false }));
      return;
    }

    if (rerootPreviewTimerRef.current !== null) {
      window.clearTimeout(rerootPreviewTimerRef.current);
      rerootPreviewTimerRef.current = null;
    }
    setRerootPreviewVisibleIds(null);
    selectNodeFromVisualNode(d, clickedNodeId);
    lastNodeClickTargetRef.current = clickedNodeId;
    setEvalActionBar(null);
    setRootPopup(null);
    setContextMenu(prev => ({ ...prev, visible: false }));
  }, [node, selectedNodePath, selectNodeFromVisualNode, onTokenSelect]);

  const closeContextMenu = useCallback(() => {
    if (!contextMenu.visible) return;

    setContextMenu(prev => ({ ...prev, visible: false }));
  }, [contextMenu.visible]);

  const handleNodeContextMenu = useCallback((event: React.MouseEvent | MouseEvent, d: HierarchyPointNodeWithData) => {
    event.preventDefault();
    event.stopPropagation();

    if (d.depth === 0 && d.data.id === 'root') {
      return;
    }

    if (!containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const relativeX = event.clientX - containerRect.left;
    const relativeY = event.clientY - containerRect.top;

    if (contextMenu.visible && contextMenu.nodeData && contextMenu.nodeData.id === d.data.id) {
      setContextMenu(prev => ({ ...prev, visible: false }));
      return;
    }

    setEvalActionBar(null);
    setRootPopup(null);
    setContextMenu({
      visible: true,
      position: { x: relativeX, y: relativeY },
      nodeData: d.data
    });
  }, [contextMenu.visible, contextMenu.nodeData]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (!contextMenu.visible) return;

      const target = e.target as Element;
      const isContextMenuClick = target.closest('.context-menu-container');
      const isNodeClick = target.closest('.node') || target.closest('.node-group');

      if (!isContextMenuClick && !isNodeClick) {
        closeContextMenu();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [contextMenu.visible, closeContextMenu]);

  // 링크 생성기 메모이제이션 - Sankey/Basic 스타일 선택
  const linkGenerator = useMemo(() => {
    // 노드의 실제 너비를 계산하는 함수
    const getNodeWidth = (nodeRef: any) => {
      const nodeData = nodeRef?.data ?? nodeRef;
      // Overview 모드에서 필터링된 노드는 점의 실제 너비만 고려
      if (effectiveOverviewMode && nodeData.isFiltered) {
        return OVERVIEW_FILTERED_NODE_DIAMETER;
      }

      const isMergedPrefixUnderRoot =
        isEffectiveVisibleRootPrefixNode(nodeRef);

      return calculateNodeTextLayout(nodeData, undefined, {
        isRootNode: nodeRef?.depth === 0,
        isMergedPrefixUnderRoot,
        isLeafNode: !nodeRef?.children || nodeRef.children.length === 0
      }).maxVisibleLineWidth;
    };

    const getOverviewLinkPath = (
      sourceX: number,
      sourceY: number,
      targetX: number,
      targetY: number
    ) => {
      const horizontalDistance = targetX - sourceX;
      const verticalDistance = targetY - sourceY;

      if (horizontalDistance <= 8 || Math.abs(verticalDistance) < 0.5) {
        return `M ${sourceX},${sourceY} L ${targetX},${targetY}`;
      }

      const elbowX = sourceX + (horizontalDistance * 0.45);
      const directionY = verticalDistance >= 0 ? 1 : -1;
      const cornerRadius = Math.min(
        6,
        Math.abs(verticalDistance) / 2,
        horizontalDistance / 3
      );

      if (cornerRadius < 1) {
        return `M ${sourceX},${sourceY} L ${targetX},${targetY}`;
      }

      return [
        `M ${sourceX},${sourceY}`,
        `L ${elbowX - cornerRadius},${sourceY}`,
        `Q ${elbowX},${sourceY} ${elbowX},${sourceY + (directionY * cornerRadius)}`,
        `L ${elbowX},${targetY - (directionY * cornerRadius)}`,
        `Q ${elbowX},${targetY} ${elbowX + cornerRadius},${targetY}`,
        `L ${targetX},${targetY}`
      ].join(' ');
    };

    const isOverviewFilteredLink = (d: any) => (
      effectiveOverviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)
    );

    if (effectiveOverviewMode) {
      return function(d: any) {
        if (!isOverviewFilteredLink(d)) {
          if (useSankeyLinks) {
            const sourceWidth = getNodeWidth(d.source);
            const sourceX = d.source.y + sourceWidth + 16;
            const sourceY = d.source.x;
            const targetX = d.target.y - 6;
            const targetY = d.target.x;

            const horizontalDistance = targetX - sourceX;
            const midX = sourceX + horizontalDistance * 0.4;
            return `M ${sourceX},${sourceY}
                    C ${midX},${sourceY} ${midX},${targetY} ${targetX},${targetY}`;
          }

          return d3.linkHorizontal<any, any>()
            .source(link => {
              const sourceWidth = getNodeWidth(link.source);
              return [link.source.y + sourceWidth + 20, link.source.x];
            })
            .target(link => [link.target.y - 18, link.target.x])(d);
        }

        const sourceWidth = getNodeWidth(d.source);
        const sourceX = d.source.data.isFiltered
          ? d.source.y + OVERVIEW_FILTERED_NODE_RADIUS + 1
          : d.source.y + sourceWidth + 12;
        const sourceY = d.source.x;
        const targetX = d.target.data.isFiltered
          ? d.target.y - OVERVIEW_FILTERED_NODE_RADIUS - 1
          : d.target.y - 8;
        const targetY = d.target.x;

        return getOverviewLinkPath(sourceX, sourceY, targetX, targetY);
      };
    }

    if (useSankeyLinks) {
      // Sankey 스타일 - 수동 곡선
      return function(d: any) {
        const sourceWidth = getNodeWidth(d.source);
        // 필터링된 노드는 중심에서 출발
        const sourceX = effectiveOverviewMode && d.source.data.isFiltered
          ? d.source.y
          : d.source.y + sourceWidth + 16;
        const sourceY = d.source.x;
        // 필터링된 노드는 중심으로 도착
        const targetX = effectiveOverviewMode && d.target.data.isFiltered
          ? d.target.y
          : d.target.y - 6;
        const targetY = d.target.x;

        const horizontalDistance = targetX - sourceX;
        const midX = sourceX + horizontalDistance * 0.4;
        return `M ${sourceX},${sourceY}
                C ${midX},${sourceY} ${midX},${targetY} ${targetX},${targetY}`;
      };
    } else {
      // 기본 스타일 - D3의 자연스러운 곡선 (원까지만 연결)
      return d3.linkHorizontal<any, any>()
        .source(d => {
          const sourceWidth = getNodeWidth(d.source);
          // 필터링된 노드는 중심에서 출발
          const sourceX = effectiveOverviewMode && d.source.data.isFiltered
            ? d.source.y
            : d.source.y + sourceWidth + 20;
          return [sourceX, d.source.x];
        })
        .target(d => {
          // 필터링된 노드는 중심으로 도착
          const targetX = effectiveOverviewMode && d.target.data.isFiltered
            ? d.target.y
            : d.target.y - 18;
          return [targetX, d.target.x];
        });
    }
  }, [useSankeyLinks, effectiveOverviewMode, isEffectiveVisibleRootPrefixNode]);

  // 색상 스케일 메모이제이션
  const colorScale = useMemo(() => {
    return d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateBlues);
  }, []);

  // 링크 두께를 위한 스케일 - Sankey 스타일로 더 극적인 차이
const linkStrokeScale = useMemo(() => {
  return d3.scaleLinear()
    .domain([0, 1])
    .range([0, 20]); // 최소 0px에서 최대 20px로 증가
}, []);





  // 뷰포트에 노드가 있는지 확인하고 없으면 루트로 이동
  useEffect(() => {
    if (!hierarchyData || !zoomRef.current || !transformRef.current || !svgRef.current) return;

    // 다음 프레임에서 뷰포트 가시성 확인 (인위적인 100ms 지연 제거)
    const rafId = requestAnimationFrame(() => {
      if (!svgRef.current) return;
      const svg = d3.select(svgRef.current);
      const nodes = hierarchyData.descendants();
      const transform = transformRef.current;

      if (!transform) return;

      // 현재 뷰포트 영역 계산
      const viewportLeft = -transform.x / transform.k;
      const viewportTop = -transform.y / transform.k;
      const viewportRight = viewportLeft + dimensions.width / transform.k;
      const viewportBottom = viewportTop + dimensions.height / transform.k;

      // 뷰포트에 보이는 노드가 있는지 확인
      const hasVisibleNodes = nodes.some(node => {
        const nodeX = node.y;
        const nodeY = node.x;
        return nodeX >= viewportLeft - 100 &&
               nodeX <= viewportRight + 100 &&
               nodeY >= viewportTop - 50 &&
               nodeY <= viewportBottom + 50;
      });

      // 보이는 노드가 없으면 루트로 이동
      if (!hasVisibleNodes && nodes.length > 0 && zoomRef.current) {
        const resetTransform = d3.zoomIdentity
          .translate(ROOT_VIEW_MARGIN.left, ROOT_VIEW_MARGIN.top)
          .scale(transform.k); // 줌 레벨은 유지

        transformRef.current = resetTransform;
        svg.transition()
          .duration(750)
          .ease(d3.easeCubicInOut)
          .call(zoomRef.current!.transform as any, resetTransform);
      }
    });

    return () => cancelAnimationFrame(rafId);
  }, [hierarchyData, dimensions]);

  // 효율적인 트리 데이터 업데이트
  useEffect(() => {
    if (!hierarchyData) return;

    // 트리 데이터와 차원 업데이트
    const containerWidth = containerRef.current?.clientWidth || 800;
    const containerHeight = containerRef.current?.clientHeight || 600;
    const treeDimensions = calculateTreeDimensions(
      hierarchyData,
      containerWidth,
      containerHeight
    );

    // 이전과 다를 때만 업데이트하여 불필요한 리렌더링 방지
    setDimensions(prev => {
      if (prev.width !== treeDimensions.width || prev.height !== treeDimensions.height) {
        return treeDimensions;
      }
      return prev;
    });

    // 트리 데이터 업데이트
    setTreeData(hierarchyData);
  }, [hierarchyData]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !tooltipRef.current) return;

    if (initializedRef.current) return;
    initializedRef.current = true;

    const svg = d3.select(svgRef.current);

    if (!svg.select('.zoom-container').node()) {
      svg.append('g')
        .attr('class', 'zoom-container')
        .append('g')
        .attr('class', 'links');

      svg.select('.zoom-container')
        .append('g')
        .attr('class', 'nodes');
    }

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.01, 4])
      .on('zoom', (event) => {
        transformRef.current = event.transform;
        svg.select('.zoom-container').attr('transform', event.transform);
      });

    zoomRef.current = zoom;

    const initialTransform = d3.zoomIdentity
      .translate(ROOT_VIEW_MARGIN.left, ROOT_VIEW_MARGIN.top)
      .scale(0.8);

    svg.call(zoom);
    svg.call(zoom.transform, initialTransform);

    transformRef.current = initialTransform;

    const handleResize = () => {
      if (hierarchyData) {
        const containerWidth = containerRef.current?.clientWidth || 800;
        const containerHeight = containerRef.current?.clientHeight || 600;
        setDimensions(calculateTreeDimensions(hierarchyData, containerWidth, containerHeight));
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [closeContextMenu, hierarchyData]);


  // 노드 개수 변경 핸들러
  const handleNodeCountChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    // 변경 전에 현재 뷰포트 중심 저장
    if (svgRef.current && transformRef.current) {
      const transform = transformRef.current;
      const width = dimensions.width;
      const height = dimensions.height;

      viewportCenterRef.current = {
        x: (width / 2 - transform.x) / transform.k,
        y: (height / 2 - transform.y) / transform.k
      };
    }

    const value = clampVisibleNodes(Number(e.target.value));
    setMaxVisibleNodes(value);

    if (node) {
      resetUserFolding(node);
    }
  }, [node, resetUserFolding, dimensions]);

  // +/- 버튼으로 노드 개수를 1개씩 조정
  const handleDiscreteNodeCountChange = useCallback((direction: 'up' | 'down') => {
    // 변경 전에 현재 뷰포트 중심 저장
    if (svgRef.current && transformRef.current) {
      const transform = transformRef.current;
      const width = dimensions.width;
      const height = dimensions.height;

      viewportCenterRef.current = {
        x: (width / 2 - transform.x) / transform.k,
        y: (height / 2 - transform.y) / transform.k
      };
    }

    let newValue;
    if (direction === 'up') {
      // 1개씩 증가 (최대 50)
      newValue = Math.min(maxVisibleNodes + 1, MAX_VISIBLE_NODES);
    } else {
      // 1개씩 감소 (최소 0)
      newValue = Math.max(maxVisibleNodes - 1, MIN_VISIBLE_NODES);
    }

    if (newValue === maxVisibleNodes) return;
    setMaxVisibleNodes(newValue);

    if (node) {
      resetUserFolding(node);
    }
  }, [maxVisibleNodes, node, resetUserFolding, dimensions]);

  const handleSeparateBigToken = useCallback((nodeData: any, clickedTokenIndex?: number) => {
    if (!node) return;
    if (nodeData.mergedNodes && nodeData.mergedNodes.nodeIds) {

      if (clickedTokenIndex !== undefined) {
        const clickedNodeId = nodeData.mergedNodes.nodeIds[clickedTokenIndex];
        const originalNode = findNodeById(node, clickedNodeId);
        if (originalNode) {
          originalNode.isExpanded = true;
        }
      } else {
        for (const id of nodeData.mergedNodes.nodeIds) {
          const originalNode = findNodeById(node, id);
          if (originalNode) {
            originalNode.isExpanded = true;
          }
        }
      }

      setTreeUpdateTrigger(prev => prev + 1);
    }
  }, [node]);

  const handleMergeToBigToken = useCallback((nodeData: any) => {
    if (!node) return;

    const currentNode = findNodeById(node, nodeData.id);
    if (!currentNode) return;

    if (currentNode.isExpanded) {
      currentNode.isExpanded = false;
    }

    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  const applyFoldNodeState = useCallback((targetTree: ExtendedVisualNode | null, nodeId: string): boolean => {
    if (!targetTree) return false;

    const targetNode = findNodeById(targetTree, nodeId);
    if (!targetNode) return false;

    targetNode.isUserFolded = true;
    targetNode.isFolded = true;
    return true;
  }, []);

  const handleFoldNode = useCallback((nodeId: string) => {
    const activeTree = latestTreeRef.current ?? node;
    if (!applyFoldNodeState(activeTree, nodeId)) return;

    setTreeUpdateTrigger(prev => prev + 1);
  }, [applyFoldNodeState, node]);

  const handleDirectLeftFoldNode = useCallback((nodeId: string) => {
    const activeTree = latestTreeRef.current ?? node;
    if (!activeTree) return;
    const targetNode = findNodeById(activeTree, nodeId);
    if (!targetNode) return;

    const existingTimerId = directLeftFoldTimerRef.current.get(nodeId);
    if (existingTimerId !== undefined) {
      window.clearTimeout(existingTimerId);
    }

    directLeftFoldHiddenIdsRef.current.set(nodeId, collectSubtreeNodeIds(targetNode));
    syncDirectLeftFoldHiddenNodeIds();

    const timerId = window.setTimeout(() => {
      directLeftFoldTimerRef.current.delete(nodeId);
      directLeftFoldHiddenIdsRef.current.delete(nodeId);
      syncDirectLeftFoldHiddenNodeIds();

      const latestActiveTree = latestTreeRef.current ?? node;
      if (!applyFoldNodeState(latestActiveTree, nodeId)) return;
      setTreeUpdateTrigger(prev => prev + 1);
    }, DIRECT_LEFT_FOLD_DELAY_MS);

    directLeftFoldTimerRef.current.set(nodeId, timerId);
  }, [applyFoldNodeState, collectSubtreeNodeIds, node, syncDirectLeftFoldHiddenNodeIds]);

  const handleFilteredNodeClick = useCallback((nodeId: string) => {
    if (!node) return;

    const targetNode = findNodeById(node, nodeId);
    if (!targetNode) return;

    targetNode.isUserFolded = false;
    targetNode.isFolded = false;

    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  const handleUnfoldAdjacentSiblings = useCallback((nodeId: string) => {
    if (!node) return;

    const currentNode = findNodeById(node, nodeId);
    if (!currentNode) return;

    const parentNode = findParentNode(node, nodeId);
    if (!parentNode || !parentNode.children) return;

    const orderedSiblings = orderChildrenByTopNRank(
      [...parentNode.children],
      topNRankByNodeId,
      centerTopNNodes
    );

    const currentIndex = orderedSiblings.findIndex(child => child.id === nodeId);
    if (currentIndex === -1) return;

    setTimeout(() => {
      for (let i = currentIndex + 1; i < orderedSiblings.length; i++) {
        const sibling = orderedSiblings[i];
        if (sibling.isFolded || sibling.isUserFolded) {
          sibling.isUserFolded = false;
          sibling.isFolded = false;
        } else {
          break;
        }
      }

      setTreeUpdateTrigger(prev => prev + 1);
    }, 500);
  }, [node, topNRankByNodeId, centerTopNNodes]);

  const handleRevealMergedHiddenBranch = useCallback((nodeId: string) => {
    if (!node) return;

    const branchNode = findNodeById(node, nodeId);
    if (!branchNode || !branchNode.children || branchNode.children.length === 0) return;

    const orderedChildren = orderChildrenByTopNRank(
      [...branchNode.children],
      topNRankByNodeId,
      centerTopNNodes
    );

    const restoredPendingExpansion = resetPendingMergeReleaseExpansion();

    const nextVisibleChild = orderedChildren.find(child =>
      child.isUserFolded !== true && child.isFolded !== true
    ) || orderedChildren.find(child => child.isUserFolded !== true)
      || orderedChildren[0];

    if (nextVisibleChild) {
      const previousIsExpanded = nextVisibleChild.isExpanded;
      if (previousIsExpanded !== true) {
        nextVisibleChild.isExpanded = true;
        mergeReleaseExpandedNodeRef.current = {
          nodeId: nextVisibleChild.id,
          previousIsExpanded
        };
      }
      triggerMergeReleaseTransition();
      setTreeUpdateTrigger(prev => prev + 1);
    }

    mergeReleaseExpandTimerRef.current = window.setTimeout(() => {
      mergeReleaseExpandTimerRef.current = null;
      let hasChanges = restoreTemporaryMergeReleaseExpansion() || restoredPendingExpansion;
      const activeTree = latestTreeRef.current;
      const activeBranchNode = activeTree ? findNodeById(activeTree, nodeId) : null;
      if (activeBranchNode?.children) {
        for (const child of activeBranchNode.children) {
          if (child.isFolded || child.isUserFolded === true) {
            child.isFolded = false;
            child.isUserFolded = false;
            hasChanges = true;
          }
        }
      }

      if (hasChanges) {
        setTreeUpdateTrigger(prev => prev + 1);
      }
    }, MERGE_RELEASE_HINT_SUPPRESS_MS);
  }, [node, resetPendingMergeReleaseExpansion, restoreTemporaryMergeReleaseExpansion, triggerMergeReleaseTransition, topNRankByNodeId, centerTopNNodes]);

  // 시각화 렌더링 콜백
  const renderVisualizationCallback = useCallback(() => {
    renderVisualization({
      svgRef,
      containerRef,
      hierarchyData,
      dimensions,
      node,
      selectedNodePath,
      useSankeyLinks,
      linkGenerator,
      colorScale,
      linkStrokeScale,
      transformRef,
      zoomRef,
      previousRootRef,
      explicitResetRef,
      viewportCenterRef,
      tooltipRef,
      handleNodeClick,
      handleNodeContextMenu,
      handleMergedRootTokenClick,
      handleRerootButtonAnchorHover,
      toggleFold,
      onTokenSelect,
      handleExpandSelectedLeafText,
      isRerootBlockedNode,
      handleSeparateBigToken,
      handleDirectLeftFoldNode,
      handleUnfoldAdjacentSiblings,
      handleRevealMergedHiddenBranch,
      handleFilteredNodeClick,
      contextMenu,
      closeContextMenu,
      effectiveVisibleRootId,
      effectiveVisibleRootPathNodeIds,
      overviewMode: effectiveOverviewMode,
      bigTokenEnabled,
      evaluationColorEnabled,
      rerootPreviewVisibleIds,
      rerootHoverPreviewTargetNodeId,
      transientHiddenNodeIds,
      mergeReleaseTransitionActive: Date.now() < mergeReleaseTransitionUntilRef.current,
      mergeReleaseHintSuppressActive: Date.now() < mergeReleaseHintSuppressUntilRef.current
    });
  }, [dimensions, hierarchyData, selectedNodePath, node, toggleFold,
    handleNodeClick, linkGenerator, colorScale, linkStrokeScale,
    useSankeyLinks, onTokenSelect, handleExpandSelectedLeafText,
    isRerootBlockedNode, handleSeparateBigToken, handleDirectLeftFoldNode, handleUnfoldAdjacentSiblings, handleRevealMergedHiddenBranch, handleFilteredNodeClick, contextMenu, closeContextMenu, effectiveVisibleRootId, effectiveVisibleRootPathNodeIds, effectiveOverviewMode, bigTokenEnabled, evaluationColorEnabled, rerootPreviewVisibleIds, rerootHoverPreviewTargetNodeId, transientHiddenNodeIds, handleNodeContextMenu, handleMergedRootTokenClick, handleRerootButtonAnchorHover]);

  // 선택된 경로가 완전히 렌더링되지 않으면 선택 해제
  useEffect(() => {
    if (!hierarchyData || selectedNodePath.length === 0) {
      return;
    }

    const renderedNodeIds = new Set<string>();
    for (const renderedNode of hierarchyData.descendants()) {
      renderedNodeIds.add(renderedNode.data.id);
      if (renderedNode.data.mergedNodes?.nodeIds) {
        for (const mergedNodeId of renderedNode.data.mergedNodes.nodeIds) {
          renderedNodeIds.add(mergedNodeId);
        }
      }
    }

    // Top-N 서브루트 모드에서는 서브루트 이전 prefix는 렌더링 대상에서 제외
    const pathToValidate = (() => {
      if (effectiveVisibleRootId === 'root') return selectedNodePath;
      const startIndex = selectedNodePath.indexOf(effectiveVisibleRootId);
      if (startIndex < 0) return selectedNodePath;
      return selectedNodePath.slice(startIndex);
    })();

    const isFullPathRendered = pathToValidate.every(id => renderedNodeIds.has(id));

    if (!isFullPathRendered) {
      if (onTokenSelect) {
        onTokenSelect('', -1);
      }
    }
  }, [hierarchyData, selectedNodePath, onTokenSelect, effectiveVisibleRootId]);

  // D3 렌더링
  useEffect(() => {
    renderVisualizationCallback();
  }, [treeData, selectedNodePath, dimensions, renderVisualizationCallback]);

  useEffect(() => {
    if (selectedNodePath.length === 0 || !svgRef.current) {
      setRootPopup(null);
      setRerootHoverPreviewTargetNodeId(null);
      return;
    }

    const selectedTerminalNodeId = selectedNodePath[selectedNodePath.length - 1];
    const selectedNodeContainer = d3.select(svgRef.current)
      .selectAll<SVGGElement, unknown>('.node-container')
      .filter(function() {
        if (this.getAttribute('data-node-id') === selectedTerminalNodeId) return true;
        const representedNodeIds = (this.getAttribute('data-represented-node-ids') || '')
          .split(',')
          .filter(Boolean);
        return representedNodeIds.includes(selectedTerminalNodeId);
      })
      .node();

    if (!selectedNodeContainer) {
      setRootPopup(null);
      return;
    }

    const representedNodeIds = (selectedNodeContainer.getAttribute('data-represented-node-ids') || '')
      .split(',')
      .filter(Boolean);
    const isSelectedVisibleRootPrefix = effectiveVisibleRootId !== 'root'
      && representedNodeIds.includes(effectiveVisibleRootId);
    const defaultBranchNodeIds = isSelectedVisibleRootPrefix
      ? new Set(effectiveVisibleRootPathNodeIds)
      : new Set(collectActualSingleBranchNodeIds(representedNodeIds[0] ?? selectedTerminalNodeId));
    const defaultTargetNodeId = isSelectedVisibleRootPrefix
      ? effectiveVisibleRootId
      : (representedNodeIds.find(nodeId => defaultBranchNodeIds.has(nodeId))
        ?? representedNodeIds[0]
        ?? selectedTerminalNodeId);
    const defaultAnchorElements = Array.from(selectedNodeContainer.querySelectorAll('.merged-token'))
      .filter((element): element is Element => {
        const tokenIndex = Number(element.getAttribute('data-token-index') ?? '-1');
        const representedNodeId = representedNodeIds[tokenIndex];
        return Boolean(representedNodeId && defaultBranchNodeIds.has(representedNodeId));
      });
    const fallbackAnchorElement = selectedNodeContainer.querySelector('.node-token-text')
      ?? selectedNodeContainer;

    setRootPopupFromAnchors(defaultTargetNodeId, defaultAnchorElements, fallbackAnchorElement);
  }, [selectedNodePath, hierarchyData, treeData, dimensions, setRootPopupFromAnchors, collectActualSingleBranchNodeIds, effectiveVisibleRootId, effectiveVisibleRootPathNodeIds]);

  useEffect(() => {
    if (!rootPopup?.visible) {
      setRerootHoverPreviewTargetNodeId(null);
    }
  }, [rootPopup]);

  // 선택된 경로의 가장 깊은 노드 위치를 상위로 보고
  useEffect(() => {
    if (!onLeafPositionChange) return;
    if (selectedNodePath.length === 0 || !svgRef.current || !containerRef.current) {
      onLeafPositionChange(null);
      return;
    }

    // 약간의 딜레이를 줘서 D3 렌더링이 완료된 후 위치를 계산
    const timerId = window.setTimeout(() => {
      if (!svgRef.current || !containerRef.current) return;

      const deepestNodeId = selectedNodePath[selectedNodePath.length - 1];

      // 먼저 정확한 nodeId로 찾고, 없으면 mergedNodes에 포함된 노드 탐색
      let nodeElement = d3.select(svgRef.current)
        .selectAll(`[data-node-id="${deepestNodeId}"]`)
        .node() as SVGGElement | null;

      if (!nodeElement && hierarchyData) {
        // mergedNodes에 포함된 경우
        for (const desc of hierarchyData.descendants()) {
          if (desc.data.mergedNodes?.nodeIds?.includes(deepestNodeId)) {
            nodeElement = d3.select(svgRef.current)
              .selectAll(`[data-node-id="${desc.data.id}"]`)
              .node() as SVGGElement | null;
            break;
          }
        }
      }

      if (nodeElement) {
        const bbox = nodeElement.getBoundingClientRect();
        const containerRect = containerRef.current!.getBoundingClientRect();
        onLeafPositionChange({
          x: bbox.right - containerRect.left,
          y: bbox.top - containerRect.top + bbox.height / 2,
          nodeId: deepestNodeId,
        });
      } else {
        onLeafPositionChange(null);
      }
    }, 50);

    return () => window.clearTimeout(timerId);
  }, [selectedNodePath, onLeafPositionChange, hierarchyData, treeData, dimensions]);
  return (
    <TreeContainer ref={containerRef}>
      {showLayoutToggles && systemSettings?.toggleVisibility === 'all' && (
        <ToggleControl>
          <AlignmentButton
            isActive={leftAligned}
            onClick={() => setLeftAligned(!leftAligned)}
            title={leftAligned ? "Switch to Top-Down Layout" : "Switch to Left-Aligned Layout"}
          >
            {leftAligned ? 'L' : 'T'}
          </AlignmentButton>

          <SankeyButton
            isActive={bigTokenEnabled}
            onClick={() => setBigTokenEnabled(!bigTokenEnabled)}
            title={bigTokenEnabled ? "Disable Big Tokens" : "Enable Big Tokens"}
          >
            {bigTokenEnabled ? 'M' : 'I'}
          </SankeyButton>

          <SankeyButton
            isActive={useStandardTree}
            onClick={() => setUseStandardTree(!useStandardTree)}
            title={useStandardTree ? "Use Custom Tree Layout" : "Use Standard Tree Layout"}
          >
            {useStandardTree ? 'S' : 'C'}
          </SankeyButton>

          <SankeyButton
            isActive={effectiveOverviewMode}
            onClick={() => setOverviewMode(!overviewMode)}
            title={overviewMode ? "Detailed View" : "Overview Mode"}
          >
            O
          </SankeyButton>

          {showCenterTopNToggle && (
            <SankeyButton
              isActive={centerTopNNodes}
              onClick={() => setCenterTopNNodes(!centerTopNNodes)}
              title={centerTopNNodes
                ? "Keep sibling order by default"
                : "Center higher Top-N ranks within each sibling block"}
            >
              N
            </SankeyButton>
          )}
        </ToggleControl>
      )}

      {showLayoutToggles && systemSettings?.toggleVisibility === 'overview-only' && (
        <ZoomControl style={{ top: '16px', width: '56px' }}>
          <SankeyButton
            isActive={effectiveOverviewMode}
            onClick={() => setOverviewMode(!overviewMode)}
            title={overviewMode ? "Detailed View" : "Overview Mode"}
            style={{ width: '42px', height: '42px' }}
          >
            O
          </SankeyButton>
        </ZoomControl>
      )}

      <ZoomControl>
        <HomeButton
          onClick={handleResetToGlobalRoot}
          title="Reset root"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#333" strokeWidth="2">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
          </svg>
        </HomeButton>
        <VerticalSliderContainer>
          <SliderText
            style={{ fontSize: '10px', color: '#888', userSelect: 'none' }}
            title="Top N mode"
          >
            Top N
          </SliderText>
          <SliderText style={{ fontSize: '11px' }}>
            {`Top ${maxVisibleNodes}`}
          </SliderText>
          <SliderButton
            onClick={() => handleDiscreteNodeCountChange('up')}
            title="Show more nodes"
          >
            +
          </SliderButton>
          <SliderWithTicks>
            <VerticalSlider
              type="range"
              min={MIN_VISIBLE_NODES}
              max={MAX_VISIBLE_NODES}
              value={clampVisibleNodes(maxVisibleNodes)}
              onChange={handleNodeCountChange}
              step="1"
            />
            <SliderTicks>
              {discreteNodeCounts.map((count, index) => {
                const range = MAX_VISIBLE_NODES - MIN_VISIBLE_NODES;
                const position = range === 0
                  ? 100
                  : ((count - MIN_VISIBLE_NODES) / range) * 100;
                return (
                  <SliderTick
                    key={index}
                    style={{
                      position: 'absolute',
                      top: `${100 - position}%`,
                      left: '0'
                    }}
                  />
                );
              })}
            </SliderTicks>
          </SliderWithTicks>
          <SliderButton
            onClick={() => handleDiscreteNodeCountChange('down')}
            title="Show fewer nodes"
          >
            -
          </SliderButton>
        </VerticalSliderContainer>
      </ZoomControl>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        onClick={(e) => {
          const targetEl = e.target as Element;

          if (targetEl.tagName === 'svg' || targetEl.classList.contains('zoom-container')) {
            if (onTokenSelect) {
              onTokenSelect('', -1);
            }
            lastNodeClickTargetRef.current = null;
            closeContextMenu();
            setEvalActionBar(null);
            setRootPopup(null);
            setRerootHoverPreviewTargetNodeId(null);
          }
        }}
      />
      <Tooltip ref={tooltipRef} />
      {markNotifications.map(notification => (
        <MarkNotification
          key={notification.id}
          x={notification.x}
          y={notification.y}
          category={notification.category}
        >
          {notification.category === 'good' ? '👍 Marked Good' : '👎 Marked Bad'}
        </MarkNotification>
      ))}
      {evalActionBar && evalActionBar.visible && (
        <EvalActionBar
          x={evalActionBar.x}
          y={evalActionBar.y}
          className="eval-action-bar"
          onClick={(e) => e.stopPropagation()}
        >
          <EvalActionButton
            variant="good"
            isActive={evalActionBar.nodeData.evaluationCategory === 'good'}
            disabled={!!evalActionBar.nodeData.ancestorEvaluation}
            onClick={(e) => {
              e.stopPropagation();
              if (evalActionBar.nodeData.ancestorEvaluation) return;
              handleNodeEvaluate(evalActionBar.nodeId, 'good');
              setEvalActionBar(null);
            }}
          >
            <span style={{ color: '#4caf50' }}>✓</span>
            {evalActionBar.nodeData.evaluationCategory === 'good' ? 'Unmark' : 'Good'}
          </EvalActionButton>
          <EvalActionButton
            variant="bad"
            isActive={evalActionBar.nodeData.evaluationCategory === 'bad'}
            disabled={!!evalActionBar.nodeData.ancestorEvaluation}
            onClick={(e) => {
              e.stopPropagation();
              if (evalActionBar.nodeData.ancestorEvaluation) return;
              handleNodeEvaluate(evalActionBar.nodeId, 'bad');
              setEvalActionBar(null);
            }}
          >
            <span style={{ color: '#f44336' }}>✗</span>
            {evalActionBar.nodeData.evaluationCategory === 'bad' ? 'Unmark' : 'Bad'}
          </EvalActionButton>
        </EvalActionBar>
      )}
      {rootPopup && rootPopup.visible && (
        <EvalActionBar
          x={rootPopup.x}
          y={rootPopup.y}
          onClick={(e) => e.stopPropagation()}
        >
          <EvalActionButton
            variant="root"
            disabled={isRerootBlockedNode(rootPopup.targetNodeId)}
            onMouseEnter={() => {
              if (isRerootBlockedNode(rootPopup.targetNodeId)) return;
              setRerootHoverPreviewTargetNodeId(rootPopup.targetNodeId);
            }}
            onMouseLeave={() => {
              setRerootHoverPreviewTargetNodeId(null);
            }}
            onClick={(e) => {
              e.stopPropagation();
              if (!node) return;
              const targetNodeId = rootPopup.targetNodeId;
              if (isRerootBlockedNode(targetNodeId)) return;
              setRerootHoverPreviewTargetNodeId(null);
              if (targetNodeId === node.id) {
                handleResetToGlobalRoot();
              } else {
                const rightmostNode = findRightmostSinglePathNode(node, targetNodeId);
                applyRerootWithPreview(targetNodeId, rightmostNode?.id ?? targetNodeId);
                if (onTokenSelect) {
                  onTokenSelect('', -1);
                }
                lastNodeClickTargetRef.current = null;
                setEvalActionBar(null);
                setRootPopup(null);
              }
            }}
          >
            <span style={{ color: '#1976d2' }}>⌂</span>
            Root
          </EvalActionButton>
        </EvalActionBar>
      )}
      <NodeContextMenu
        position={contextMenu.position}
        nodeData={contextMenu.nodeData}
        treeNode={node}
        onPinNode={onNodePin}
        onEvaluateNode={handleNodeEvaluate}
        onGenerateFromNode={handleGenerateFromNode}
        onSeparateBigToken={handleSeparateBigToken}
        onMergeToBigToken={handleMergeToBigToken}
        onClose={closeContextMenu}
        onFoldNode={handleFoldNode}
        isVisible={contextMenu.visible}
        isGenerating={contextMenu.nodeData ? contextMenu.nodeData.isGenerating : false}
        isCompleted={contextMenu.nodeData ? contextMenu.nodeData.isCompleted : false}
        isBigToken={contextMenu.nodeData && !!contextMenu.nodeData.mergedNodes}
        isExpanded={contextMenu.nodeData ? contextMenu.nodeData.isExpanded : false}
      />
    </TreeContainer>
  );
};


export const TokenTreeVisualizerMemo = React.memo(TokenTreeVisualizer);
export default TokenTreeVisualizerMemo;
