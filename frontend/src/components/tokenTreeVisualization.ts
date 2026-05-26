import * as d3 from 'd3';
import { findNodeById, findParentNode } from '../utils/treeTransform';
import {
  calculateNodeTextLayout,
  findPathToNode,
  NODE_MAX_LINES,
  NODE_LINE_HEIGHT,
  NODE_VERTICAL_PADDING,
  HierarchyPointNodeWithData,
  OVERVIEW_FILTERED_NODE_RADIUS
} from './tokenTreeUtils';
import {
  hideTooltip,
  showTooltip,
  addFoldButton,
  renderPinnedIndicator
} from './tokenTreeInteractions';

interface RenderVisualizationParams {
  svgRef: React.RefObject<SVGSVGElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  hierarchyData: HierarchyPointNodeWithData | null;
  dimensions: { width: number; height: number };
  node: any;
  selectedNodePath: string[];
  useSankeyLinks: boolean;
  linkGenerator: any;
  colorScale: d3.ScaleSequential<string>;
  linkStrokeScale: d3.ScaleLinear<number, number>;
  transformRef: React.MutableRefObject<d3.ZoomTransform | null>;
  zoomRef: React.MutableRefObject<d3.ZoomBehavior<SVGSVGElement, unknown> | null>;
  previousRootRef: React.MutableRefObject<any>;
  explicitResetRef: React.MutableRefObject<boolean>;
  viewportCenterRef: React.MutableRefObject<{ x: number; y: number } | null>;
  tooltipRef: React.RefObject<HTMLDivElement | null>;
  handleNodeClick: (event: React.MouseEvent | MouseEvent, d: HierarchyPointNodeWithData) => void;
  handleNodeContextMenu: (event: React.MouseEvent | MouseEvent, d: HierarchyPointNodeWithData) => void;
  handleMergedRootTokenClick: (event: React.MouseEvent | MouseEvent, d: HierarchyPointNodeWithData, tokenIndex: number) => void;
  handleRerootButtonAnchorHover?: (
    targetNodeId: string,
    anchorElements?: ArrayLike<Element> | null,
    fallbackAnchorElement?: Element | null
  ) => void;
  toggleFold: (nodeId: string) => void;
  onTokenSelect?: (sequenceId: string, tokenIndex: number) => void;
  handleExpandSelectedLeafText: (nodeId: string) => void;
  isRerootBlockedNode?: (nodeId: string) => boolean;
  handleSeparateBigToken: (nodeData: any, clickedTokenIndex?: number) => void;
  handleDirectLeftFoldNode: (nodeId: string) => void;
  handleUnfoldAdjacentSiblings: (nodeId: string) => void;
  handleRevealMergedHiddenBranch: (nodeId: string) => void;
  handleFilteredNodeClick?: (nodeId: string) => void;
  contextMenu: {
    visible: boolean;
    position: { x: number, y: number };
    nodeData: any;
  };
  closeContextMenu: () => void;
  effectiveVisibleRootId: string;
  effectiveVisibleRootPathNodeIds: string[];
  unfoldPathToNode?: (nodeId: string) => void;
  overviewMode?: boolean;
  bigTokenEnabled?: boolean;
  evaluationColorEnabled?: boolean;
  rerootPreviewVisibleIds?: Set<string> | null;
  rerootHoverPreviewTargetNodeId?: string | null;
  transientHiddenNodeIds?: Set<string>;
  mergeReleaseTransitionActive?: boolean;
  mergeReleaseHintSuppressActive?: boolean;
}

const ROOT_VIEW_MARGIN = { top: 70, right: 120, bottom: 20, left: 90 };
const DEFAULT_SANKEY_COLOR = '#9ec9f2';
const OVERVIEW_LINK_STROKE_WIDTH = 4;
const OVERVIEW_FILTERED_NODE_HIT_RADIUS = 10;
const OVERVIEW_FILTERED_NODE_HOVER_RADIUS = OVERVIEW_FILTERED_NODE_RADIUS + 3;

export const renderVisualization = (params: RenderVisualizationParams) => {
  const {
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
    overviewMode = false,
    bigTokenEnabled = true,
    evaluationColorEnabled = true,
    rerootPreviewVisibleIds = null,
    rerootHoverPreviewTargetNodeId = null,
    transientHiddenNodeIds = new Set<string>(),
    mergeReleaseTransitionActive = false,
    mergeReleaseHintSuppressActive = false
  } = params;

  if (!svgRef.current || !containerRef.current || !hierarchyData) return;

  const svg = d3.select(svgRef.current);
  const g = svg.select<SVGGElement>('.zoom-container');

  // viewBox 대신 width/height만 설정
  svg
    .attr('width', dimensions.width)
    .attr('height', dimensions.height);

  // 뷰 리셋 필요한지 확인 - 간소화된 로직
  const shouldResetView = (() => {
    // 명시적 리셋 요청(Home 버튼)이 있거나, 최초 렌더링인 경우만 리셋
    return explicitResetRef.current || !previousRootRef.current;
  })();

  // 현재 노드를 previousRootRef에 저장
  previousRootRef.current = node;

  // 뷰 리셋 또는 현재 transform 유지
  if (shouldResetView && zoomRef.current) {
    const resetTransform = d3.zoomIdentity
      .translate(ROOT_VIEW_MARGIN.left, ROOT_VIEW_MARGIN.top)
      .scale(0.8);

    transformRef.current = resetTransform;

    svg.transition()
      .duration(750)
      .call(zoomRef.current.transform, resetTransform);
  } else if (transformRef.current) {
    // 뷰포트 중심점이 저장되어 있으면 복원
    if (viewportCenterRef.current && zoomRef.current) {
      const currentScale = transformRef.current.k;
      const newX = dimensions.width / 2 - viewportCenterRef.current.x * currentScale;
      const newY = dimensions.height / 2 - viewportCenterRef.current.y * currentScale;

      const newTransform = d3.zoomIdentity
        .translate(newX, newY)
        .scale(currentScale);

      transformRef.current = newTransform;
      svg.call(zoomRef.current.transform, newTransform);

      // 복원 후 초기화
      viewportCenterRef.current = null;
    } else {
      // 현재 transform 유지 (transition 없이 즉시 적용)
      g.attr('transform', transformRef.current.toString());
    }
  }

  const isNodeVisibleDuringRerootPreview = (nodeData: any): boolean => {
    if (!rerootPreviewVisibleIds || rerootPreviewVisibleIds.size === 0) return true;
    if (rerootPreviewVisibleIds.has(nodeData.id)) return true;
    const mergedNodeIds = nodeData?.mergedNodes?.nodeIds as string[] | undefined;
    return Boolean(mergedNodeIds?.some(nodeId => rerootPreviewVisibleIds.has(nodeId)));
  };

  const isNodeTemporarilyHidden = (nodeData: any): boolean => {
    if (!transientHiddenNodeIds || transientHiddenNodeIds.size === 0) return false;
    if (transientHiddenNodeIds.has(nodeData.id)) return true;
    const mergedNodeIds = nodeData?.mergedNodes?.nodeIds as string[] | undefined;
    return Boolean(mergedNodeIds?.some(nodeId => transientHiddenNodeIds.has(nodeId)));
  };

  const isNodeIncludedInPreview = (nodeData: any, previewVisibleIds: Set<string>): boolean => {
    if (previewVisibleIds.has(nodeData.id)) return true;
    const mergedNodeIds = nodeData?.mergedNodes?.nodeIds as string[] | undefined;
    return Boolean(mergedNodeIds?.some(nodeId => previewVisibleIds.has(nodeId)));
  };

  const collectPreviewVisibleIds = (targetNodeId: string): Set<string> | null => {
    const targetNode = findNodeById(node, targetNodeId);
    if (!targetNode) return null;

    const previewVisibleIds = new Set<string>();
    const stack = [targetNode];
    while (stack.length > 0) {
      const currentNode = stack.pop();
      if (!currentNode) continue;
      previewVisibleIds.add(currentNode.id);
      currentNode.children.forEach(child => stack.push(child));
    }

    findPathToNode(node, targetNodeId).forEach(pathNode => {
      previewVisibleIds.add(pathNode.id);
    });

    return previewVisibleIds;
  };

  // 링크 데이터 조인
  const links = hierarchyData.links();
  const defaultTransitionDuration = 500;
  const mergeReleaseTransitionDuration = 180;
  const linkTransitionDuration = mergeReleaseTransitionActive
    ? mergeReleaseTransitionDuration
    : defaultTransitionDuration;
  const nodeTransitionDuration = mergeReleaseTransitionActive
    ? mergeReleaseTransitionDuration
    : defaultTransitionDuration;

  const linkSelection = g.select('.links')
    .selectAll<SVGPathElement, any>('path')
    .data(links, (d: any) => `${d.source.data.id}-${d.target.data.id}`);

  linkSelection.exit().remove();

  const getCollapsedLinkPath = (linkDatum: any): string | null => (
    linkGenerator({
      ...linkDatum,
      target: linkDatum.source
    })
  );

  const linkEnter = linkSelection.enter()
    .append('path')
    .attr('class', 'link')
    .attr('fill', 'none')
    .attr('stroke-linecap', 'butt')
    .attr('opacity', 1)
    .attr('d', d => mergeReleaseTransitionActive ? getCollapsedLinkPath(d) : linkGenerator(d));

  const allLinks = linkEnter.merge(linkSelection);

  allLinks
    .attr('stroke', d => {
      const sourceId = d.source.data.id;
      const targetId = d.target.data.id;
      const targetNode = d.target.data;
      const hasOverviewPrunedBadDescendant = Boolean(targetNode.overviewDepthPrunedHasBadDescendant);

      // 선택된 경로는 항상 우선적으로 파란색으로 표시
      if (selectedNodePath.includes(sourceId) && selectedNodePath.includes(targetId)) {
        return '#2196f3'; // 파란색
      }

      // 대상 노드의 평가 상태에 따라 색상 결정
      // 직접 평가된 경우 우선, ancestorEvaluation이 있으면 그것 사용
      const evaluationStatus = targetNode.evaluationCategory || targetNode.ancestorEvaluation;

      // Overview depth cutoff 아래 bad descendant가 있으면 링크도 빨간색 우선
      if (evaluationColorEnabled && hasOverviewPrunedBadDescendant) {
        return '#e57373';
      }

      // Overview 모드에서 필터링된 노드도 평가 상태에 따라 색상 적용
      if (evaluationColorEnabled && evaluationStatus === 'good') {
        // Good으로 평가된 링크는 녹색 (필터링된 경우 연한 녹색)
        if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
          return '#81c784'; // 연한 녹색
        }
        return '#4caf50'; // 진한 녹색
      } else if (evaluationColorEnabled && evaluationStatus === 'bad') {
        // Bad로 평가된 링크는 빨간색 (필터링된 경우 연한 빨간색)
        if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
          return '#e57373'; // 연한 빨간색
        }
        return '#f44336'; // 진한 빨간색
      }

      // 평가되지 않은 필터링된 노드는 회색 (더 진하게)
      if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
        return DEFAULT_SANKEY_COLOR;
      }

      // 일반 링크는 하늘색 (선택 경로 대비를 위해 더 연하게)
      return DEFAULT_SANKEY_COLOR;
    })
    .attr('stroke-linejoin', overviewMode ? 'round' : 'miter')
    .attr('stroke-width', d => {
      if (overviewMode) {
        return OVERVIEW_LINK_STROKE_WIDTH;
      }

      if (useSankeyLinks) {
        // Sankey 스타일: 확률 기반 두께
        const baseWidth = linkStrokeScale(d.target.data.prob);
        // Overview 모드에서 필터링된 노드는 1/4 두께
        if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
          return baseWidth * 0.25;
        }
        return baseWidth;
      } else {
        // 기본 스타일: 얇은 링크
        return 2;
      }
    })
    .attr('opacity', d => {
      // 필터링된 링크는 모두 동일하게 반투명
      if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
        return 0.7;
      }
      return 1;
    })
    .style('visibility', d => (
      isNodeVisibleDuringRerootPreview(d.source.data) &&
      isNodeVisibleDuringRerootPreview(d.target.data) &&
      !isNodeTemporarilyHidden(d.source.data) &&
      !isNodeTemporarilyHidden(d.target.data)
    ) ? 'visible' : 'hidden')
    .style('pointer-events', d => (
      isNodeVisibleDuringRerootPreview(d.source.data) &&
      isNodeVisibleDuringRerootPreview(d.target.data) &&
      !isNodeTemporarilyHidden(d.source.data) &&
      !isNodeTemporarilyHidden(d.target.data)
    ) ? 'auto' : 'none')
    .on('mouseover', (event, d) => showTooltip(event, d.target, tooltipRef, containerRef))
    .on('mousemove', (event, d) => showTooltip(event, d.target, tooltipRef, containerRef))
    .on('mouseout', () => hideTooltip(tooltipRef));

  // 경로만 부드럽게 전환
  allLinks
    .transition()
    .duration(linkTransitionDuration)
    .ease(d3.easeCubicOut)
    .attr('d', linkGenerator);

  // 노드 렌더링
  // Overview 모드에서는 모든 노드를 렌더링 (필터링된 노드는 점으로)
  const nodesToRender = hierarchyData.descendants();
  const renderedNodeIdSet = new Set(nodesToRender.map(nodeItem => nodeItem.data.id));
  const renderedNodeDataById = new Map(
    nodesToRender.map(nodeItem => [nodeItem.data.id, nodeItem.data] as const)
  );
  const selectedTerminalNodeId = selectedNodePath.length > 0
    ? selectedNodePath[selectedNodePath.length - 1]
    : null;
  const collectActualSingleBranchNodeIds = (startNodeId: string): string[] => {
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
      if (!nextNode || isRerootBlockedNode?.(nextNode.id)) break;
      downwardNode = nextNode;
      branchNodeIds.push(downwardNode.id);
    }

    return branchNodeIds;
  };

    const nodeSelection = g.select('.nodes')
      .selectAll<SVGGElement, any>('.node')
      .data(nodesToRender, (d: any) => d.data.id);

    nodeSelection.exit().remove();

  const getNodeTargetTransform = (d: any): string => `translate(${d.y},${d.x})`;
  const getNodeEnterTransform = (d: any): string => {
    if (!mergeReleaseTransitionActive || !d.parent) return getNodeTargetTransform(d);
    return `translate(${d.parent.y},${d.parent.x})`;
  };

    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'node')
      .attr('data-node-id', d => d.data.id)
      .attr('transform', d => getNodeEnterTransform(d));

    if (mergeReleaseTransitionActive) {
      nodeEnter
        .transition()
        .duration(nodeTransitionDuration)
        .ease(d3.easeCubicOut)
        .attr('transform', d => getNodeTargetTransform(d));
    }

    nodeSelection
      .attr('data-node-id', d => d.data.id)
      .transition()
      .duration(nodeTransitionDuration)
      .ease(d3.easeCubicOut)
      .attr('transform', d => getNodeTargetTransform(d));

  const allNodes = nodeEnter.merge(nodeSelection);
  allNodes
    .style('visibility', d => (
      isNodeVisibleDuringRerootPreview(d.data) && !isNodeTemporarilyHidden(d.data)
    ) ? 'visible' : 'hidden')
    .style('pointer-events', d => (
      isNodeVisibleDuringRerootPreview(d.data) && !isNodeTemporarilyHidden(d.data)
    ) ? 'auto' : 'none')
    .style('opacity', 1);

    const clearHoverRerootDimming = () => {
      allLinks.attr('opacity', 1);
      allNodes.style('opacity', 1);
    };

    const applyHoverRerootDimming = (targetNodeId: string) => {
      const previewVisibleIds = collectPreviewVisibleIds(targetNodeId);
      if (!previewVisibleIds) {
        clearHoverRerootDimming();
        return;
      }

      allLinks.attr('opacity', d => {
        const isCurrentlyVisible = (
          isNodeVisibleDuringRerootPreview(d.source.data) &&
          isNodeVisibleDuringRerootPreview(d.target.data) &&
          !isNodeTemporarilyHidden(d.source.data) &&
          !isNodeTemporarilyHidden(d.target.data)
        );
        if (!isCurrentlyVisible) return 1;

        return (
          isNodeIncludedInPreview(d.source.data, previewVisibleIds) &&
          isNodeIncludedInPreview(d.target.data, previewVisibleIds)
        ) ? 1 : 0.3;
      });

      allNodes.style('opacity', d => {
        const isCurrentlyVisible = (
          isNodeVisibleDuringRerootPreview(d.data) &&
          !isNodeTemporarilyHidden(d.data)
        );
        if (!isCurrentlyVisible) return 1;
        return isNodeIncludedInPreview(d.data, previewVisibleIds) ? 1 : 0.3;
      });
    };

    if (rerootHoverPreviewTargetNodeId) {
      applyHoverRerootDimming(rerootHoverPreviewTargetNodeId);
    } else {
      clearHoverRerootDimming();
    }

    allNodes.each(function (d) {
    const container = d3.select(this);

    container.selectAll('*').remove();

    if (overviewMode && d.data.isFiltered) {
      const evaluationStatus = d.data.evaluationCategory || d.data.ancestorEvaluation;
      const hasOverviewPrunedBadDescendant = Boolean(d.data.overviewDepthPrunedHasBadDescendant);

      let fillColor = DEFAULT_SANKEY_COLOR;
      let strokeColor = DEFAULT_SANKEY_COLOR;

      if (evaluationColorEnabled && hasOverviewPrunedBadDescendant) {
        fillColor = '#e57373';
        strokeColor = '#f44336';
      } else if (evaluationColorEnabled && evaluationStatus === 'good') {
        fillColor = '#81c784';
        strokeColor = '#4caf50';
      } else if (evaluationColorEnabled && evaluationStatus === 'bad') {
        fillColor = '#e57373';
        strokeColor = '#f44336';
      }

      const hitArea = container.append('circle')
        .attr('r', OVERVIEW_FILTERED_NODE_HIT_RADIUS)
        .attr('fill', 'transparent')
        .attr('stroke', 'none')
        .style('cursor', 'pointer');

      const filteredNodeCircle = container.append('circle')
        .attr('r', OVERVIEW_FILTERED_NODE_RADIUS)
        .attr('fill', fillColor)
        .attr('stroke', strokeColor)
        .attr('stroke-width', 1)
        .attr('opacity', 1)
        .style('pointer-events', 'none');

      if (handleFilteredNodeClick) {
        hitArea.on('click', function(event: MouseEvent) {
          if (event.button !== 0) {
            return;
          }

          event.stopPropagation();
          handleFilteredNodeClick(d.data.id);
        })
        .on('contextmenu', function(event: MouseEvent) {
          handleNodeContextMenu(event, d);
        });

        hitArea
          .on('mouseover', function() {
            filteredNodeCircle
              .attr('r', OVERVIEW_FILTERED_NODE_HOVER_RADIUS)
              .attr('stroke-width', 1.5);
          })
          .on('mouseout', function() {
            filteredNodeCircle
              .attr('r', OVERVIEW_FILTERED_NODE_RADIUS)
              .attr('stroke-width', 1);
          });
      }

      return; // 점만 그리고 종료
    }

    const nodeContainer = container.append('g')
      .attr('class', 'node-container')
      .attr('data-node-id', d.data.mergedNodes && d.data.mergedNodes.nodeIds
        ? d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1]
        : d.data.id);

    // 노드 텍스트 레이아웃 계산 (tokenTreeUtils의 단일 로직 사용)
    const isTextExpanded = Boolean(d.data.isTextExpanded);
    const isRootResetKey = d.depth === 0 && d.data.id === 'root';
    const originalNode = findNodeById(node, d.data.id);

    let hasChildren = false;
    let mergedLastNodeId: string | null = null;
    const isRenderedLeaf = !d.children || d.children.length === 0;

    if (overviewMode) {
      hasChildren = (d.data.children?.length ?? 0) > 0;
    } else if (d.data.mergedNodes && d.data.mergedNodes.nodeIds && d.data.mergedNodes.nodeIds.length > 0) {
      // 병합된 노드의 마지막 노드 ID 가져오기
      mergedLastNodeId = d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1];
      // 마지막 노드로 자식 존재 여부 체크
      const lastNode = findNodeById(node, mergedLastNodeId);
      hasChildren = lastNode ? (lastNode.children && lastNode.children.length > 0) : false;
    } else {
      // 일반 노드는 그대로 체크
      hasChildren = originalNode ? (originalNode.children && originalNode.children.length > 0) : false;
    }

    const representedNodeIds = (d.data.mergedNodes?.nodeIds && d.data.mergedNodes.nodeIds.length > 0)
      ? d.data.mergedNodes.nodeIds
      : [d.data.id];
    const isMergedPrefixUnderRoot = effectiveVisibleRootId !== 'root'
      && d.depth === 1
      && d.parent?.depth === 0
      && representedNodeIds.includes(effectiveVisibleRootId);
    const leafTextToggleNodeId = mergedLastNodeId ?? d.data.id;
    const textLayout = calculateNodeTextLayout(d.data, undefined, {
      isRootNode: d.depth === 0,
      isMergedPrefixUnderRoot,
      isLeafNode: isRenderedLeaf
    });
    const collapsedTextLayout = calculateNodeTextLayout(d.data, NODE_MAX_LINES, {
      isRootNode: d.depth === 0,
      isMergedPrefixUnderRoot,
      isLeafNode: isRenderedLeaf
    });
    const textWidth = textLayout.maxVisibleLineWidth;

    // 리프 노드는 우측 아이콘이 없으므로 오른쪽 여백을 줄여 폭을 컴팩트하게 유지
    const totalWidth = textWidth + (hasChildren ? 20 : 8);
    const nodeHeight = textLayout.height;
    const collapsedNodeHeight = collapsedTextLayout.height;
    const nodeTopY = isTextExpanded ? -collapsedNodeHeight / 2 : -nodeHeight / 2;
    const verticalExpandOffset = isTextExpanded ? (nodeHeight - collapsedNodeHeight) / 2 : 0;
    const isPathSelected = representedNodeIds.some(nodeId => selectedNodePath.includes(nodeId));
    const isDirectlySelected = selectedTerminalNodeId !== null && representedNodeIds.includes(selectedTerminalNodeId);
    const showExpandLeafButton = isDirectlySelected && isRenderedLeaf && !isRootResetKey;
    const isLeafTextManuallyExpanded = d.data.forcedTextMaxLines === 6;
    const canExpandLeafTextFurther = showExpandLeafButton
      && isTextExpanded
      && textLayout.truncated
      && !isLeafTextManuallyExpanded;
    const canCollapseLeafText = showExpandLeafButton && isLeafTextManuallyExpanded;
    const canToggleLeafTextExpansion = canExpandLeafTextFurther || canCollapseLeafText;
    nodeContainer.attr('data-represented-node-ids', representedNodeIds.join(','));
    if (isTextExpanded || showExpandLeafButton) {
      container.raise();
    }
    const isPinned = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        // 병합된 노드인 경우: 모든 노드 중 하나라도 고정되어 있으면 true
        return d.data.mergedNodes.nodeIds.some(nodeId => {
          const targetNode = findNodeById(node, nodeId);
          return targetNode?.isPinned || false;
        });
      } else {
        // 일반 노드인 경우
        return Boolean(d.data.isPinned);
      }
    })();

    const isEvaluated = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        // 병합된 노드인 경우: 모든 노드 중 하나라도 평가되어 있으면 true
        return d.data.mergedNodes.nodeIds.some(nodeId => {
          const targetNode = findNodeById(node, nodeId);
          return targetNode?.isEvaluated || false;
        });
      } else {
        // 일반 노드인 경우
        return Boolean(d.data.isEvaluated);
      }
    })();

    const evaluationCategory = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        // 병합된 노드인 경우: 제일 부모(첫 번째) 평가된 노드의 카테고리 사용
        for (const nodeId of d.data.mergedNodes.nodeIds) {
          const targetNode = findNodeById(node, nodeId);
          if (targetNode?.evaluationCategory) {
            return targetNode.evaluationCategory;
          }
        }
        return null;
      } else {
        // 일반 노드인 경우
        return d.data.evaluationCategory || null;
      }
    })();

    const ancestorEvaluation = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        // 병합된 노드인 경우: 직접 평가된 노드가 있으면 상속은 무시
        const hasDirectEvaluation = d.data.mergedNodes.nodeIds.some(nodeId => {
          const targetNode = findNodeById(node, nodeId);
          return targetNode?.evaluationCategory;
        });

        if (hasDirectEvaluation) {
          // 직접 평가된 노드가 있으면 ancestorEvaluation 무시
          return null;
        }

        // 직접 평가된 노드가 없을 때만 ancestorEvaluation 확인
        for (const nodeId of d.data.mergedNodes.nodeIds) {
          const targetNode = findNodeById(node, nodeId);
          if (targetNode?.ancestorEvaluation) {
            return targetNode.ancestorEvaluation;
          }
        }
        return null;
      } else {
        // 일반 노드인 경우
        return d.data.ancestorEvaluation || null;
      }
    })();


    const isGenerating = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        // 병합된 노드인 경우: 마지막 노드의 생성 상태 확인
        const lastNodeId = d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1];
        const lastNode = findNodeById(node, lastNodeId);
        return lastNode?.nodeState === 'generating' || false;
      } else {
        return d.data.nodeState === 'generating';
      }
    })();

    const isCompleted = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        const lastNodeId = d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1];
        const lastNode = findNodeById(node, lastNodeId);
        return lastNode?.nodeState === 'completed' || false;
      } else {
        return d.data.nodeState === 'completed';
      }
    })();

    d.data.isGenerating = isGenerating;
    d.data.isCompleted = isCompleted;
    d.data.isEvaluated = isEvaluated;
    d.data.evaluationCategory = evaluationCategory;
    d.data.ancestorEvaluation = ancestorEvaluation;

    const nodeGroup = nodeContainer.append('g')
      .attr('class', 'node-group');

    const hasOverviewPrunedBadDescendant = Boolean(d.data.overviewDepthPrunedHasBadDescendant);
    const hasGoodEvaluation = evaluationColorEnabled
      && !hasOverviewPrunedBadDescendant
      && (evaluationCategory === 'good' || ancestorEvaluation === 'good');
    const hasBadEvaluation = evaluationColorEnabled
      && (
        hasOverviewPrunedBadDescendant
        || evaluationCategory === 'bad'
        || ancestorEvaluation === 'bad'
      );
    const baseFill = isPinned ? '#fff8e1' : 'white';
    const baseStroke = (() => {
      if (hasGoodEvaluation) return '#4caf50';
      if (hasBadEvaluation) return '#f44336';
      if (isPathSelected) return '#2196f3';
      if (isPinned) return '#ffb74d';
      return '#c5ced6';
    })();
    const baseStrokeWidth = (hasGoodEvaluation || hasBadEvaluation)
      ? 3
      : (isPathSelected || isPinned ? 2 : 1);
    const effectiveBaseStrokeWidth = d.data.mergedNodes ? Math.max(baseStrokeWidth, 2) : baseStrokeWidth;
    const hoverStroke = (() => {
      if (hasGoodEvaluation) return '#388e3c';
      if (hasBadEvaluation) return '#c62828';
      return '#2196f3';
    })();
    const hoverStrokeWidth = Math.max(effectiveBaseStrokeWidth, 2);
    const shouldHighlightOnHover = !isDirectlySelected;

    const background = nodeGroup.append('rect')
      .attr('x', -6)
      .attr('y', nodeTopY)
      .attr('width', totalWidth + 12)
      .attr('height', nodeHeight)
      .attr('fill', baseFill)
      .attr('stroke', baseStroke)
      .attr('stroke-width', effectiveBaseStrokeWidth)
      .attr('rx', 6)
      .style('opacity', 1)
      .style('cursor', 'pointer');

    if (isMergedPrefixUnderRoot) {
      nodeGroup.append('line')
        .attr('x1', totalWidth + 1.5)
        .attr('x2', totalWidth + 1.5)
        .attr('y1', nodeTopY + 6)
        .attr('y2', nodeTopY + nodeHeight - 6)
        .attr('stroke', '#90a4ae')
        .attr('stroke-width', 2)
        .attr('stroke-linecap', 'round')
        .attr('opacity', isDirectlySelected ? 0.9 : 0.72)
        .style('pointer-events', 'none');
    }

    const handleMouseOver = function (event: any) {
      if (shouldHighlightOnHover) {
        background
          .attr('fill', baseFill)
          .attr('stroke', hoverStroke)
          .attr('stroke-width', hoverStrokeWidth)
          .style('filter', hasGoodEvaluation || hasBadEvaluation ? 'none' : 'drop-shadow(0px 0px 3px rgba(33, 150, 243, 0.25))');
      }
    };

    const handleMouseOut = function () {
      background
        .attr('fill', baseFill)
        .attr('stroke', baseStroke)
        .attr('stroke-width', effectiveBaseStrokeWidth)
        .style('filter', 'none');
      hideTooltip(tooltipRef);
    };

    background
      .on('mouseover', handleMouseOver)
      .on('mouseout', handleMouseOut)
      .on('click', (event) => handleNodeClick(event, d))
      .on('contextmenu', (event) => handleNodeContextMenu(event, d));

    if (!useSankeyLinks && d.data.prob !== undefined && d.data.prob > 0) {
      nodeGroup.append('circle')
        .attr('cx', -12)
        .attr('cy', 0)
        .attr('r', 6)
        .attr('fill', colorScale(d.data.prob))
        .attr('stroke', '#333')
        .attr('stroke-width', 1)
        .attr('opacity', 1);
    }

    const textElement = nodeGroup.append('text')
      .attr('font-weight', 'normal')
      .attr('dominant-baseline', 'middle');
    if (isRootResetKey) {
      textElement
        .attr('x', totalWidth / 2)
        .attr('y', 0)
        .attr('text-anchor', 'middle')
        .attr('font-size', 18)
        .attr('font-weight', 700)
        .attr('fill', '#1976d2')
        .style('pointer-events', 'none')
        .text('⌂');
    } else if (isTextExpanded) {
      textElement.attr('y', nodeTopY + (NODE_VERTICAL_PADDING / 2) + (NODE_LINE_HEIGHT / 2));
    } else {
      textElement.attr('y', -(textLayout.visibleLineCount - 1) * (NODE_LINE_HEIGHT / 2));
    }
    if (!isRootResetKey) {
      textElement.attr('class', 'node-token-text');
    }
    const mergedTokenSpanNodes = new Map<number, SVGTSpanElement[]>();
    const collectMergedTokenAnchorElements = (startNodeId: string, fallbackTokenIndex: number): Element[] => {
      const actualSingleBranchNodeIdSet = new Set(collectActualSingleBranchNodeIds(startNodeId));
      const anchorElements: Element[] = [];

      representedNodeIds.forEach((representedNodeId, representedTokenIndex) => {
        if (!actualSingleBranchNodeIdSet.has(representedNodeId)) return;
        const tokenSpans = mergedTokenSpanNodes.get(representedTokenIndex);
        if (!tokenSpans || tokenSpans.length === 0) return;
        anchorElements.push(...tokenSpans);
      });

      if (anchorElements.length > 0) {
        return anchorElements;
      }

      return [...(mergedTokenSpanNodes.get(fallbackTokenIndex) ?? [])];
    };
    const setActualSingleBranchHighlight = (fill: string | null, startNodeId: string = representedNodeIds[0]) => {
      if (!svgRef.current) return;

      const svgSelection = d3.select(svgRef.current);
      if (fill === null) {
        svgSelection
          .selectAll<SVGTextElement | SVGTSpanElement, unknown>('.node-token-text, .node-token-text tspan')
          .attr('fill', null);
        return;
      }

      const actualSingleBranchNodeIdSet = new Set(collectActualSingleBranchNodeIds(startNodeId));
      if (actualSingleBranchNodeIdSet.size === 0) return;

      svgSelection
        .selectAll<SVGGElement, unknown>('.node-container')
        .each(function() {
          const tokenContainer = d3.select(this);
          const representedIds = (tokenContainer.attr('data-represented-node-ids') || '')
            .split(',')
            .filter(Boolean);
          const mergedTokenSelection = tokenContainer.selectAll<SVGTSpanElement, unknown>('.merged-token');

          if (mergedTokenSelection.size() > 0 && representedIds.length > 0) {
            tokenContainer
              .selectAll<SVGTextElement, unknown>('.node-token-text')
              .attr('fill', null);
            mergedTokenSelection
              .attr('fill', function() {
                const tokenIndex = Number(d3.select(this).attr('data-token-index') ?? '-1');
                const nodeId = representedIds[tokenIndex];
                return nodeId && actualSingleBranchNodeIdSet.has(nodeId)
                  ? fill
                  : null;
              });
            return;
          }

          const shouldHighlight = representedIds.some(nodeId => actualSingleBranchNodeIdSet.has(nodeId));

          tokenContainer
            .selectAll<SVGTextElement | SVGTSpanElement, unknown>('.node-token-text, .node-token-text tspan')
            .attr('fill', shouldHighlight ? fill : null);
        });
    };

    // 병합된 노드인 경우 개별 토큰으로 분리하여 표시
    if (isRootResetKey) {
      // root는 텍스트 대신 아이콘만 렌더링한다.
    } else if (d.data.mergedNodes && d.data.mergedNodes.tokens) {
      textLayout.lines.forEach((line, lineIndex) => {
        line.forEach((item, fragmentIndex) => {
          const tokenIndex = item.tokenIndex ?? 0;
          const tokenNodeId = d.data.mergedNodes?.nodeIds?.[tokenIndex]
            ?? representedNodeIds[tokenIndex]
            ?? representedNodeIds[0];
          const popupTargetNodeId = tokenNodeId;
          const popupFallbackTokenIndex = tokenIndex;
          const isMergedTokenInteractionEnabled =
            isMergedPrefixUnderRoot || isDirectlySelected;
          const isDisabledLeafTerminalToken = isDirectlySelected
            && Boolean(popupTargetNodeId && isRerootBlockedNode?.(popupTargetNodeId));

          const wordSpan = textElement.append('tspan')
            .text(item.text)
            .style('cursor', (isMergedTokenInteractionEnabled && !isDisabledLeafTerminalToken) ? 'pointer' : 'inherit')
            .style('pointer-events', (isMergedTokenInteractionEnabled && !isDisabledLeafTerminalToken) ? 'auto' : 'none')
            .attr('class', 'merged-token')
            .attr('data-token-index', tokenIndex);

          if (fragmentIndex === 0) {
            wordSpan
              .attr('x', 2)
              .attr('dy', lineIndex === 0 ? 0 : `${NODE_LINE_HEIGHT}px`);
          }

          const wordSpanNode = wordSpan.node();
          if (wordSpanNode) {
            const existingTokenSpans = mergedTokenSpanNodes.get(tokenIndex);
            if (existingTokenSpans) {
              existingTokenSpans.push(wordSpanNode);
            } else {
              mergedTokenSpanNodes.set(tokenIndex, [wordSpanNode]);
            }
          }

          if (!isMergedTokenInteractionEnabled) {
            return;
          }

          // 호버 효과
          wordSpan
            .on('mouseover', function(event) {
              const hoveredNodeId = tokenNodeId;
              if (isDirectlySelected) {
                setActualSingleBranchHighlight('#2196f3', hoveredNodeId);
                handleMouseOver(event);
                if (handleRerootButtonAnchorHover) {
                  handleRerootButtonAnchorHover(
                    popupTargetNodeId,
                    collectMergedTokenAnchorElements(popupTargetNodeId, popupFallbackTokenIndex),
                    this as SVGTextContentElement
                  );
                }
                return;
              }

              if (isMergedPrefixUnderRoot) {
                return;
              }

              d3.select(this).attr('fill', '#2196f3');
            })
            .on('mouseout', function() {
              if (isDirectlySelected) {
                setActualSingleBranchHighlight(null);
                handleMouseOut();
                return;
              }

              if (isMergedPrefixUnderRoot) {
                return;
              }

              d3.select(this).attr('fill', null);
            })
            .on('click', function(event) {
              if (event.button !== 0) {
                return;
              }

              event.stopPropagation();
              event.stopImmediatePropagation();
              if (isDirectlySelected) {
                handleRerootButtonAnchorHover?.(
                  popupTargetNodeId,
                  collectMergedTokenAnchorElements(popupTargetNodeId, popupFallbackTokenIndex),
                  this as SVGTextContentElement
                );
                return;
              }

              // 컨텍스트 메뉴가 열려있고 현재 노드의 메뉴일 때만 분리 허용
              if (!contextMenu.visible || contextMenu.nodeData?.id !== d.data.id) {
                // 일반 클릭은 병합 토큰 핸들러에서 선택/루트변경 단계적으로 처리
                handleMergedRootTokenClick(event, d, tokenIndex);
                return;
              }

              // 병합된 노드 분리 후 해당 토큰 선택
              const targetNodeId = d.data.mergedNodes?.nodeIds?.[tokenIndex];
              handleSeparateBigToken(d.data, tokenIndex);

              // 분리 후 해당 노드 선택
              if (targetNodeId && onTokenSelect) {
                // 약간의 지연을 주어 트리가 업데이트된 후 선택
                setTimeout(() => {
                  const tempSequenceId = `temp-${targetNodeId}`;
                  onTokenSelect(tempSequenceId, 0);
                }, 10);
              }

              // 컨텍스트 메뉴 닫기
              closeContextMenu();

              // 툴팁 숨기기
              hideTooltip(tooltipRef);
            });
        });
      });

      if (textLayout.truncated) {
        textElement.append('tspan').text('...');
      }
    } else {
      const isSingleTokenInteractionEnabled = isDirectlySelected
        && !Boolean(isRerootBlockedNode?.(d.data.id));
      textLayout.lines.forEach((line, lineIndex) => {
        line.forEach((item, fragmentIndex) => {
          const wordSpan = textElement.append('tspan')
            .text(item.text)
            .style('cursor', isSingleTokenInteractionEnabled ? 'pointer' : 'inherit')
            .style('pointer-events', isSingleTokenInteractionEnabled ? 'auto' : 'none');

          if (fragmentIndex === 0) {
            wordSpan
              .attr('x', 2)
              .attr('dy', lineIndex === 0 ? 0 : `${NODE_LINE_HEIGHT}px`);
          }

          const wordSpanNode = wordSpan.node();
          if (wordSpanNode) {
            const existingTokenSpans = mergedTokenSpanNodes.get(0);
            if (existingTokenSpans) {
              existingTokenSpans.push(wordSpanNode);
            } else {
              mergedTokenSpanNodes.set(0, [wordSpanNode]);
            }
          }

          if (!isSingleTokenInteractionEnabled) {
            return;
          }

          wordSpan
            .on('mouseover', function(event) {
              setActualSingleBranchHighlight('#2196f3', d.data.id);
              handleMouseOver(event);
              handleRerootButtonAnchorHover?.(
                d.data.id,
                mergedTokenSpanNodes.get(0),
                this as SVGTextContentElement
              );
            })
            .on('mouseout', function() {
              setActualSingleBranchHighlight(null);

              handleMouseOut();
            })
            .on('click', function(event) {
              if (event.button !== 0) {
                return;
              }

              event.stopPropagation();
              event.stopImmediatePropagation();
              handleRerootButtonAnchorHover?.(
                d.data.id,
                mergedTokenSpanNodes.get(0),
                this as SVGTextContentElement
              );
            })
            .on('contextmenu', function(event) {
              event.preventDefault();
              event.stopPropagation();
              handleNodeContextMenu(event, d);
            });

        });
      });

      if (textLayout.truncated) {
        textElement.append('tspan').text('...');
      }
    }

    const mergedHiddenBranchMeta = d.data.mergedNodes?.hiddenBranches;
    type HiddenHintPoint = {
      nodeId: string;
      tokenIndex: number;
      hiddenCount: number;
      source: 'child' | 'sibling';
      ownerNodeId: string;
    };

    const hiddenHintPoints: HiddenHintPoint[] = [];
    if (mergedHiddenBranchMeta?.points) {
      mergedHiddenBranchMeta.points.forEach((point) => {
        if (point.hiddenCount <= 0) return;

        // 분기 노드가 이미 별도 노드로 렌더된 경우(자체 힌트 가능),
        // 상위 merged 노드의 child 힌트는 중복으로 보이므로 숨긴다.
        const isBranchRenderedSeparately = point.nodeId !== d.data.id
          && renderedNodeIdSet.has(point.nodeId);
        if (isBranchRenderedSeparately) return;

        const branchNode = findNodeById(node, point.nodeId);
        // 마지막 토큰 분기에서, 분기 노드의 유일 visible child가 이미 렌더되어
        // 자신의 sibling 힌트를 가지는 경우 부모 child 힌트를 숨겨 중복 표시를 방지한다.
        const shouldHideBecauseVisibleChildOwnsSiblingHint = (() => {
          if (!branchNode || !branchNode.children || branchNode.children.length === 0) return false;
          const visibleChildren = branchNode.children.filter(child => child.isFolded !== true);
          if (visibleChildren.length !== 1) return false;

          const renderedVisibleChild = renderedNodeDataById.get(visibleChildren[0].id);
          if (!renderedVisibleChild) return false;

          const visibleChildSiblingHiddenCount = renderedVisibleChild.mergedNodes
            ? (renderedVisibleChild.foldedSiblingCount ?? 0)
            : (renderedVisibleChild.userFoldedSiblingCount ?? 0);
          return visibleChildSiblingHiddenCount > 0;
        })();
        if (shouldHideBecauseVisibleChildOwnsSiblingHint) return;

        const hasExpandedChild = mergeReleaseHintSuppressActive
          && Boolean(branchNode?.children?.some(child => child.isExpanded === true));
        if (hasExpandedChild) return;

        hiddenHintPoints.push({
          nodeId: point.nodeId,
          tokenIndex: point.tokenIndex,
          hiddenCount: point.hiddenCount,
          source: 'child',
          ownerNodeId: point.nodeId
        });
      });
    }

    const suppressSiblingHintDuringMergeRelease = mergeReleaseHintSuppressActive
      && Boolean(d.data.mergedNodes && d.data.isExpanded === true);
    const siblingHiddenCount = d.data.mergedNodes
      ? (d.data.foldedSiblingCount ?? 0)
      : (d.data.userFoldedSiblingCount ?? 0);
    if (siblingHiddenCount > 0 && !suppressSiblingHintDuringMergeRelease) {
      hiddenHintPoints.push({
        nodeId: d.data.id,
        tokenIndex: 0,
        hiddenCount: siblingHiddenCount,
        source: 'sibling',
        ownerNodeId: d.data.id
      });
    }

    if (hiddenHintPoints.length > 0) {
      const hiddenBranchMarkerGroup = nodeGroup.append('g')
        .attr('class', 'merged-hidden-branch-hints');

      const tokenCount = Math.max(d.data.mergedNodes?.tokens?.length || 1, 1);
      const markerMinX = 10;
      const markerMaxX = Math.max(markerMinX, textWidth);
      const rawHintScale = Math.sqrt(nodeHeight / Math.max(collapsedNodeHeight, 1));
      const hintScale = Math.max(
        1,
        Math.min(
          2.7,
          1 + ((rawHintScale - 1) * 2)
        )
      );
      const markerGap = 3;
      const markerBaseY = nodeTopY + nodeHeight + markerGap;
      const tokenCenterXByIndex = new Map<number, number>();
      const tokenWidthByIndex = new Map<number, number>();
      mergedTokenSpanNodes.forEach((spanNodes, tokenIndex) => {
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;

        const spanMetrics = spanNodes
          .map(spanNode => {
            const bbox = spanNode.getBBox();
            const textContent = spanNode.textContent || '';
            const hasRenderableWidth = bbox.width > 0.01;
            const hasNonWhitespaceText = textContent.trim().length > 0;
            return {
              bbox,
              hasRenderableWidth,
              hasNonWhitespaceText
            };
          })
          .filter(metric => metric.hasRenderableWidth || metric.hasNonWhitespaceText);

        if (spanMetrics.length === 0) return;

        // 토큰 시작의 공백 fragment는 중심 계산에서 제외해 단어 중심에 가깝게 정렬
        const firstContentIndex = spanMetrics.findIndex(metric => metric.hasNonWhitespaceText);
        const effectiveMetrics = firstContentIndex > 0
          ? spanMetrics.slice(firstContentIndex)
          : spanMetrics;
        const boundsSource = effectiveMetrics.length > 0 ? effectiveMetrics : spanMetrics;

        boundsSource.forEach(metric => {
          minX = Math.min(minX, metric.bbox.x);
          maxX = Math.max(maxX, metric.bbox.x + metric.bbox.width);
        });

        if (!Number.isFinite(minX) || !Number.isFinite(maxX) || maxX < minX) return;
        tokenCenterXByIndex.set(tokenIndex, (minX + maxX) / 2);
        tokenWidthByIndex.set(tokenIndex, maxX - minX);
      });

      const sortedTokenCenters = Array.from(tokenCenterXByIndex.entries())
        .sort((a, b) => a[0] - b[0]);

      const resolveTokenCenterX = (tokenIndex: number, fallbackX: number): number => {
        const direct = tokenCenterXByIndex.get(tokenIndex);
        if (direct !== undefined) return direct;
        if (sortedTokenCenters.length === 0) return fallbackX;

        let leftNeighbor: [number, number] | null = null;
        let rightNeighbor: [number, number] | null = null;

        for (const entry of sortedTokenCenters) {
          if (entry[0] <= tokenIndex) {
            leftNeighbor = entry;
          }
          if (entry[0] >= tokenIndex) {
            rightNeighbor = entry;
            break;
          }
        }

        if (!leftNeighbor) return rightNeighbor ? rightNeighbor[1] : fallbackX;
        if (!rightNeighbor) return leftNeighbor[1];
        if (leftNeighbor[0] === rightNeighbor[0]) return leftNeighbor[1];

        const interpolateRatio = (tokenIndex - leftNeighbor[0]) / (rightNeighbor[0] - leftNeighbor[0]);
        return leftNeighbor[1] + (rightNeighbor[1] - leftNeighbor[1]) * interpolateRatio;
      };

      const resolveHintAnchorTokenIndex = (point: HiddenHintPoint): number => {
        if (point.source === 'sibling') return point.tokenIndex;

        // child 분기는 기본적으로 "다음 토큰" 기준으로 정렬하되,
        // 다음 토큰이 없는 마지막 토큰 분기라면 현재 토큰 기준으로 폴백한다.
        const nextTokenIndex = point.tokenIndex + 1;
        if (mergedTokenSpanNodes.has(nextTokenIndex)) return nextTokenIndex;
        if (mergedTokenSpanNodes.has(point.tokenIndex)) return point.tokenIndex;
        return nextTokenIndex;
      };

      const escapeHtml = (value: string): string => (
        value
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;')
      );

      const buildHiddenTokenPreviewHtml = (hintPoint: HiddenHintPoint): string => {
        const candidateSource = (() => {
          if (hintPoint.source === 'sibling') {
            const parentNode = findParentNode(node, hintPoint.ownerNodeId);
            if (!parentNode || !parentNode.children || parentNode.children.length === 0) {
              return null;
            }
            return {
              parentId: parentNode.id,
              fallbackSelectedId: hintPoint.ownerNodeId,
              candidates: [...parentNode.children].sort((a, b) => b.prob - a.prob)
            };
          }

          const branchNode = findNodeById(node, hintPoint.nodeId);
          if (!branchNode || !branchNode.children || branchNode.children.length === 0) {
            return null;
          }
          return {
            parentId: branchNode.id,
            fallbackSelectedId: null,
            candidates: [...branchNode.children].sort((a, b) => b.prob - a.prob)
          };
        })();

        if (!candidateSource) {
          return '<div style="padding:8px 6px; color:#5f7489;">No hidden branches</div>';
        }
        const { parentId, fallbackSelectedId, candidates: allChildren } = candidateSource;

        if (allChildren.length === 0) {
          return '<div style="padding:8px 6px; color:#5f7489;">No hidden branches</div>';
        }

        const selectedPathIndex = selectedNodePath.indexOf(parentId);
        const selectedChildIdFromPath = selectedPathIndex >= 0
          ? selectedNodePath[selectedPathIndex + 1]
          : null;
        const selectedChildId = selectedChildIdFromPath && allChildren.some(child => child.id === selectedChildIdFromPath)
          ? selectedChildIdFromPath
          : (allChildren.find(child => child.id === fallbackSelectedId)?.id
            ?? allChildren.find(child => !child.isFolded && child.isUserFolded !== true)?.id
            ?? null);

        const maxItems = 6;
        const itemsHtml = allChildren.slice(0, maxItems).map(child => {
          const isSelected = child.id === selectedChildId;
          return `
          <div style="
            display:flex;
            align-items:center;
            padding:6px 6px;
            border-bottom:1px solid #e4ebf3;
            background:${isSelected ? '#dcecff' : 'transparent'};
            border-left:2px solid ${isSelected ? '#3b8ddb' : 'transparent'};
          ">
            <span style="
              flex:1;
              min-width:0;
              overflow:hidden;
              text-overflow:ellipsis;
              white-space:nowrap;
              color:#213447;
              font-weight:${isSelected ? 'bold' : 'normal'};
            ">${escapeHtml(child.token || '')}</span>
          </div>
          `;
        }).join('');

        const remaining = allChildren.length - maxItems;
        const moreHtml = remaining > 0
          ? `<div style="padding:6px 6px; color:#4a6077; font-size:11px;">+${remaining} more</div>`
          : '';

        return `
          <div style="min-width:96px; max-width:140px;">
            ${itemsHtml}
            ${moreHtml}
          </div>
        `;
      };

      hiddenHintPoints.forEach((point) => {
        const anchorTokenIndex = resolveHintAnchorTokenIndex(point);
        const anchorRatio = tokenCount > 1 ? (anchorTokenIndex / (tokenCount - 1)) : 0.5;
        const clampedAnchorRatio = Math.max(0, Math.min(1, anchorRatio));
        const fallbackAnchorX = markerMinX + (markerMaxX - markerMinX) * clampedAnchorRatio;
        const resolvedMarkerX = resolveTokenCenterX(anchorTokenIndex, fallbackAnchorX);
        const markerX = Math.max(markerMinX - 6, Math.min(markerMaxX + 10, resolvedMarkerX));
        const barCount = Math.max(1, point.hiddenCount);
        const barGap = 4;
        const referenceTokenWidth = tokenWidthByIndex.get(anchorTokenIndex)
          ?? tokenWidthByIndex.get(point.tokenIndex)
          ?? 0;
        const baseBarWidth = Math.max(
          10 * hintScale,
          referenceTokenWidth * (0.5 + (hintScale - 1) * 0.15)
        );
        const hoverBarWidth = Math.max(14 * hintScale, baseBarWidth + (4 * hintScale));
        const baseBarHeight = Math.max(3, 3 * hintScale);
        const hoverBarHeight = Math.max(5, 5 * hintScale);
        const hoverStackHeight = barCount * hoverBarHeight + (barCount - 1) * barGap;
        const stackStartY = 0;

        const markerGroup = hiddenBranchMarkerGroup.append('g')
          .attr('class', 'merged-hidden-branch-marker')
          .attr('transform', `translate(${markerX}, ${markerBaseY})`)
          .style('cursor', 'pointer');

        const markerBars: d3.Selection<SVGRectElement, unknown, null, undefined>[] = [];
        for (let barIndex = 0; barIndex < barCount; barIndex++) {
          const barY = stackStartY + barIndex * (baseBarHeight + barGap);
          const markerBar = markerGroup.append('rect')
            .attr('x', -baseBarWidth / 2)
            .attr('y', barY)
            .attr('width', baseBarWidth)
            .attr('height', baseBarHeight)
            .attr('rx', 1.5)
            .attr('fill', '#7fa3c2')
            .attr('opacity', 0.95);
          markerBars.push(markerBar);
        }

        let isPreviewVisible = false;

        const setBarsHovered = () => {
          markerBars.forEach((markerBar, barIndex) => {
            const hoverStartY = 0;
            const hoverY = hoverStartY + barIndex * (hoverBarHeight + barGap);
            markerBar
              .transition()
              .duration(120)
              .attr('x', -hoverBarWidth / 2)
              .attr('y', hoverY)
              .attr('width', hoverBarWidth)
              .attr('height', hoverBarHeight)
              .attr('rx', 2)
              .attr('fill', '#2196f3');
          });
        };

        const setBarsDefault = () => {
          markerBars.forEach((markerBar, barIndex) => {
            const barY = stackStartY + barIndex * (baseBarHeight + barGap);
            markerBar
              .transition()
              .duration(120)
              .attr('x', -baseBarWidth / 2)
              .attr('y', barY)
              .attr('width', baseBarWidth)
              .attr('height', baseBarHeight)
              .attr('rx', 1.5)
              .attr('fill', '#7fa3c2');
          });
        };

        const hintedTokenIndex = anchorTokenIndex;
        const setHintedTokenStyle = (active: boolean) => {
          const spans = mergedTokenSpanNodes.get(hintedTokenIndex);
          if (!spans) return;

          spans.forEach(spanNode => {
            const spanSelection = d3.select(spanNode);
            if (active) {
              spanSelection.attr('fill', '#2196f3');
            } else {
              spanSelection.attr('fill', null);
            }
          });
        };

        const positionPreview = () => {
          if (!tooltipRef.current || !containerRef.current) return;
          const tooltip = d3.select(tooltipRef.current);
          const containerRect = containerRef.current.getBoundingClientRect();
          const tooltipNode = tooltipRef.current;
          const markerRects = markerBars
            .map(markerBar => markerBar.node()?.getBoundingClientRect())
            .filter((rect): rect is DOMRect => rect !== undefined);

          if (markerRects.length === 0) return;

          const markerLeft = Math.min(...markerRects.map(rect => rect.left));
          const markerRight = Math.max(...markerRects.map(rect => rect.right));
          const markerTop = Math.min(...markerRects.map(rect => rect.top));
          const markerBottom = Math.max(...markerRects.map(rect => rect.bottom));

          const tooltipWidth = tooltipNode.offsetWidth;
          const tooltipHeight = tooltipNode.offsetHeight;
          const markerWidth = markerRight - markerLeft;

          let tooltipX = markerLeft - containerRect.left + (markerWidth - tooltipWidth) / 2;
          let tooltipY = markerBottom - containerRect.top + 8;

          if (tooltipX + tooltipWidth > containerRect.width - 8) {
            tooltipX = containerRect.width - tooltipWidth - 8;
          }
          if (tooltipY + tooltipHeight > containerRect.height - 8) {
            tooltipY = markerTop - containerRect.top - tooltipHeight - 8;
          }

          tooltipX = Math.max(8, tooltipX);
          tooltipY = Math.max(8, tooltipY);

          tooltip
            .style('left', `${tooltipX}px`)
            .style('top', `${tooltipY}px`);
        };

        const showPreview = () => {
          if (!tooltipRef.current) return;
          const tooltip = d3.select(tooltipRef.current);
          tooltip
            .style('opacity', 1)
            .style('padding', '6px 3px')
            .style('border', '1px solid #e7edf3')
            .style('box-shadow', '0 2px 6px rgba(0,0,0,0.08)')
            .html(buildHiddenTokenPreviewHtml(point));
          isPreviewVisible = true;
          positionPreview();
        };

        const resetPreviewTooltipStyle = () => {
          if (!tooltipRef.current) return;
          const tooltip = d3.select(tooltipRef.current);
          tooltip
            .style('padding', null)
            .style('border', null)
            .style('box-shadow', null);
        };

        markerGroup.append('rect')
          .attr('x', -Math.max(16 * hintScale, hoverBarWidth))
          .attr('y', -Math.max(6, 6 * hintScale))
          .attr('width', Math.max(32 * hintScale, hoverBarWidth * 2))
          .attr('height', Math.max(30 * hintScale, hoverStackHeight + (12 * hintScale)))
          .attr('fill', 'transparent')
          .on('mouseover', function(event) {
            event.stopPropagation();
            event.stopImmediatePropagation();
            background
              .attr('fill', baseFill)
              .attr('stroke', baseStroke)
              .attr('stroke-width', effectiveBaseStrokeWidth)
              .style('filter', 'none');
          })
          .on('mouseenter', function(event) {
            event.stopPropagation();
            event.stopImmediatePropagation();
            setBarsHovered();
            setHintedTokenStyle(true);
            showPreview();
          })
          .on('mousemove', function(event) {
            event.stopPropagation();
            if (isPreviewVisible) {
              positionPreview();
            }
          })
          .on('mouseout', function(event) {
            event.stopPropagation();
            event.stopImmediatePropagation();
            if (isPreviewVisible) {
              hideTooltip(tooltipRef);
              isPreviewVisible = false;
            }
            resetPreviewTooltipStyle();
            setHintedTokenStyle(false);
            setBarsDefault();
          })
          .on('click', function(event) {
            event.stopPropagation();
            event.stopImmediatePropagation();
            if (point.source === 'sibling') {
              handleUnfoldAdjacentSiblings(point.ownerNodeId);
            } else {
              handleRevealMergedHiddenBranch(point.nodeId);
            }
            if (isPreviewVisible) {
              hideTooltip(tooltipRef);
              isPreviewVisible = false;
            }
            resetPreviewTooltipStyle();
            setHintedTokenStyle(false);
          });
      });
    }

    if (isPinned) {
      renderPinnedIndicator(nodeGroup, -2, -8);
    }

    nodeGroup
      .on('mouseover', handleMouseOver)
      .on('mouseout', handleMouseOut)
      .on('click', function (event) {
        handleNodeClick(event, d);
      })
      .on('contextmenu', function(event) {
        handleNodeContextMenu(event, d);
      });

    const hasSiblings = d.parent && d.parent.children && d.parent.children.length > 1;

    const shouldShowLeftFoldButton = !isRootResetKey && (hasSiblings || !bigTokenEnabled);

    if (shouldShowLeftFoldButton) {
      const leftButtonGroup = nodeContainer.append('g')
        .attr('class', 'left-button-group')
        .attr('transform', `translate(-15, ${verticalExpandOffset})`)
        .style('opacity', 0);

      leftButtonGroup.append('rect')
        .attr('x', -12)
        .attr('y', -nodeHeight/2)
        .attr('width', 24)
        .attr('height', nodeHeight)
        .attr('fill', 'white')
        .attr('stroke', '#dee2e6')
        .attr('stroke-width', 1)
        .attr('rx', 6)
        .style('opacity', 0.95)
        .style('cursor', 'pointer')
        .on('mouseover', function() {
          d3.select(this)
            .attr('fill', '#f8f9fa')
            .attr('stroke', '#adb5bd');
        })
        .on('mouseout', function() {
          d3.select(this)
            .attr('fill', 'white')
            .attr('stroke', '#dee2e6');
        })
        .on('click', function(event) {
          event.stopPropagation();
          handleDirectLeftFoldNode(d.data.id);
        });

      leftButtonGroup.append('path')
        .attr('d', 'M -7,0 L 7,0')
        .attr('stroke', '#495057')
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .style('pointer-events', 'none');

    const leftHoverArea = nodeContainer.append('rect')
      .attr('class', 'left-hover-area')
      .attr('x', -30)
      .attr('y', nodeTopY)
      .attr('width', 30)
      .attr('height', nodeHeight)
      .attr('fill', 'transparent')
      .style('cursor', 'pointer');

    leftHoverArea
      .on('mouseenter', function() {
        leftButtonGroup
          .transition()
          .duration(150)
          .style('opacity', 1);
      })
      .on('mouseleave', function() {
        leftButtonGroup
          .transition()
          .duration(150)
          .style('opacity', 0);
      })
      .on('click', function(event) {
        event.stopPropagation();
        handleDirectLeftFoldNode(d.data.id);
      });

    leftButtonGroup
      .on('mouseenter', function() {
        d3.select(this).style('opacity', 1);
      })
      .on('mouseleave', function() {
        d3.select(this)
          .transition()
          .duration(150)
          .style('opacity', 0);
      });
    }

    const hasFoldedChildren = hasChildren && d.data.children.some((child: any) => child.isFolded === true);

    if (hasFoldedChildren && !isRootResetKey) {
      const buttonGroup = nodeContainer.append('g')
        .attr('class', 'fold-button-group')
        .attr('transform', `translate(${textWidth + 15}, ${verticalExpandOffset})`)
        .style('opacity', 0);

      addFoldButton(
        buttonGroup,
        d,
        (event) => {
          event.stopPropagation();

          if (mergedLastNodeId) {
            toggleFold(mergedLastNodeId);
          } else {
            toggleFold(d.data.id);
          }
        },
        nodeHeight
      );

      const buttonHoverArea = nodeContainer.append('rect')
        .attr('class', 'button-hover-area')
        .attr('x', textWidth + 6)
        .attr('y', nodeTopY)
        .attr('width', 28)
        .attr('height', nodeHeight)
        .attr('fill', 'transparent')
        .style('cursor', 'pointer');

      // 버튼 영역 hover 이벤트
      buttonHoverArea
        .on('mouseenter', function() {
          buttonGroup
            .transition()
            .duration(150)
            .style('opacity', 1);
        })
        .on('mouseleave', function() {
          buttonGroup
            .transition()
            .duration(150)
            .style('opacity', 0);
        })
        .on('click', function(event) {
          event.stopPropagation();

          // 병합된 노드인 경우 마지막 노드 ID를 toggleFold 함수에 직접 전달
          if (mergedLastNodeId) {
            toggleFold(mergedLastNodeId);
          } else {
            toggleFold(d.data.id);
          }
          // setTreeUpdateTrigger는 tokenTreePanel에서 처리됨
        });

      // 버튼 그룹 자체에도 hover 이벤트
      buttonGroup
        .on('mouseenter', function() {
          d3.select(this).style('opacity', 1);
        })
        .on('mouseleave', function() {
          d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 0);
        });
    }

    if (showExpandLeafButton) {
      const expandButtonSize = 34;
      const expandButtonHalfSize = expandButtonSize / 2;
      const expandButtonX = totalWidth + 16;
      const expandButtonY = verticalExpandOffset;
      const expandButtonIconPath = isLeafTextManuallyExpanded
        ? 'M -5,2 L 0,-3 L 5,2'
        : 'M -5,-2 L 0,3 L 5,-2';

      const expandButtonGroup = nodeContainer.append('g')
        .attr('class', 'expand-leaf-button-group')
        .attr('transform', `translate(${expandButtonX}, ${expandButtonY})`)
        .style('opacity', 1)
        .style('cursor', canToggleLeafTextExpansion ? 'pointer' : 'default');

      expandButtonGroup.append('rect')
        .attr('x', -expandButtonHalfSize)
        .attr('y', -expandButtonHalfSize)
        .attr('width', expandButtonSize)
        .attr('height', expandButtonSize)
        .attr('fill', 'white')
        .attr('stroke', canToggleLeafTextExpansion ? '#2196f3' : '#ced4da')
        .attr('stroke-width', canToggleLeafTextExpansion ? 1.85 : 1.15)
        .attr('rx', 8)
        .style('opacity', 0.98)
        .style('cursor', canToggleLeafTextExpansion ? 'pointer' : 'default')
        .on('mouseover', function() {
          if (!canToggleLeafTextExpansion) return;
          d3.select(this)
            .attr('fill', '#e3f2fd')
            .attr('stroke', '#1976d2');
        })
        .on('mouseout', function() {
          d3.select(this)
            .attr('fill', 'white')
            .attr('stroke', canToggleLeafTextExpansion ? '#2196f3' : '#ced4da');
        })
        .on('click', function(event) {
          event.stopPropagation();
          event.stopImmediatePropagation();
          if (!canToggleLeafTextExpansion) return;
          handleExpandSelectedLeafText(leafTextToggleNodeId);
        });

      expandButtonGroup.append('path')
        .attr('d', expandButtonIconPath)
        .attr('stroke', canToggleLeafTextExpansion ? '#1976d2' : '#adb5bd')
        .attr('stroke-width', 2.6)
        .attr('fill', 'none')
        .attr('stroke-linecap', 'round')
        .attr('stroke-linejoin', 'round')
        .style('pointer-events', 'none');
    }
  });

  // 사용자 액션 플래그 초기화
  explicitResetRef.current = false;

};
