import React, { RefObject } from 'react';
import * as d3 from 'd3';
import { ContextMenu, ContextMenuItem, ContextMenuDivider } from './tokenTreeStyles';
import { HierarchyPointNodeWithData } from './tokenTreeUtils';
import { VisualNode } from '../types/types';

interface NodeContextMenuProps {
  position: { x: number, y: number };
  nodeData: any;
  treeNode: VisualNode | null;
  onPinNode?: (nodeId: string) => void;
  onEvaluateNode?: (nodeId: string, category: 'good' | 'bad') => void;
  onGenerateFromNode?: (nodeId: string) => Promise<any>;
  onSeparateBigToken?: (nodeData: any) => void;
  onMergeToBigToken?: (nodeData: any) => void;
  onFoldNode?: (nodeId: string) => void;
  onClose: () => void;
  isVisible: boolean;
  isGenerating?: boolean;
  isCompleted?: boolean;
  isBigToken?: boolean;
  isExpanded?: boolean;
}

export const NodeContextMenu: React.FC<NodeContextMenuProps> = ({
  position,
  nodeData,
  onEvaluateNode,
  onClose,
  isVisible
}) => {
  if (!isVisible || !nodeData) return null;

  const handleMenuContainerClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const getTargetNodeId = () => {
    return nodeData.mergedNodes?.nodeIds
      ? nodeData.mergedNodes.nodeIds[nodeData.mergedNodes.nodeIds.length - 1]
      : nodeData.id;
  };

  return (
    <ContextMenu
      style={{
        left: position.x,
        top: position.y + 20,
        zIndex: 2000,
      }}
      onClick={handleMenuContainerClick}
      className="context-menu-container"
    >

      {onEvaluateNode && (
        <>
          <ContextMenuItem
            onClick={(e) => {
              e.stopPropagation();
              e.nativeEvent.stopImmediatePropagation();

              if (nodeData.ancestorEvaluation) {
                return;
              }

              if (onEvaluateNode) {
                onEvaluateNode(getTargetNodeId(), 'good');
              }
              onClose();
            }}
            style={{
              padding: '10px 16px',
              background: nodeData.ancestorEvaluation === 'good' ? '#e8f5e9' :
                         nodeData.ancestorEvaluation === 'bad' ? '#ffebee' :
                         nodeData.evaluationCategory === 'good' ? '#e8f5e9' : 'white',
              display: 'flex',
              alignItems: 'center',
              cursor: nodeData.ancestorEvaluation ? 'not-allowed' : 'pointer',
              opacity: nodeData.ancestorEvaluation ? 0.6 : 1,
              color: (nodeData.evaluationCategory === 'good' || nodeData.ancestorEvaluation === 'good') ? '#2e7d32' : 'inherit',
            }}
            className="evaluate-good-button"
          >
            <span style={{ marginRight: '8px', color: '#4caf50' }}>✓</span>
            {nodeData.ancestorEvaluation === 'good'
              ? "Inherited Good from Parent"
              : nodeData.ancestorEvaluation === 'bad'
              ? "Inherited from Parent (Cannot mark as Good)"
              : nodeData.evaluationCategory === 'good'
                ? "Unmark Good"
                : "Mark as Good"}
          </ContextMenuItem>

          <ContextMenuDivider />

          <ContextMenuItem
            onClick={(e) => {
              e.stopPropagation();
              e.nativeEvent.stopImmediatePropagation();

              if (nodeData.ancestorEvaluation) {
                return;
              }

              if (onEvaluateNode) {
                onEvaluateNode(getTargetNodeId(), 'bad');
              }
              onClose();
            }}
            style={{
              padding: '10px 16px',
              background: nodeData.ancestorEvaluation === 'bad' ? '#ffebee' :
                         nodeData.ancestorEvaluation === 'good' ? '#e8f5e9' :
                         nodeData.evaluationCategory === 'bad' ? '#ffebee' : 'white',
              display: 'flex',
              alignItems: 'center',
              cursor: nodeData.ancestorEvaluation ? 'not-allowed' : 'pointer',
              opacity: nodeData.ancestorEvaluation ? 0.6 : 1,
              color: (nodeData.evaluationCategory === 'bad' || nodeData.ancestorEvaluation === 'bad') ? '#c62828' : 'inherit',
            }}
            className="evaluate-bad-button"
          >
            <span style={{ marginRight: '8px', color: '#f44336' }}>✗</span>
            {nodeData.ancestorEvaluation === 'bad'
              ? "Inherited Bad from Parent"
              : nodeData.ancestorEvaluation === 'good'
              ? "Inherited from Parent (Cannot mark as Bad)"
              : nodeData.evaluationCategory === 'bad'
                ? "Unmark Bad"
                : "Mark as Bad"}
          </ContextMenuItem>
        </>
      )}
    </ContextMenu>
  );
};

export const showTooltip = (
  event: MouseEvent,
  d: HierarchyPointNodeWithData,
  tooltipRef: RefObject<HTMLDivElement | null>,
  containerRef: RefObject<HTMLDivElement | null>
): void => {
  if (!tooltipRef.current || !containerRef.current) return;

  const tooltip = d3.select(tooltipRef.current);
  const containerRect = containerRef.current.getBoundingClientRect();

  const relativeX = event.clientX - containerRect.left;
  const relativeY = event.clientY - containerRect.top;

  const nodeStatus = d.data.nodeState ?
    `<div>Status: ${d.data.nodeState}</div>` : '';

  const cumulativeProb = d.data.cumulativeProb || d.data.prob;

  tooltip
    .style('opacity', 1)
    .html(`
      <div style="padding: 5px;">
        <div>Probability: ${(d.data.prob * 100).toFixed(1)}%</div>
        <div>Cumulative: ${(cumulativeProb * 100).toFixed(2)}%</div>
        ${nodeStatus}
      </div>
    `);

  const tooltipNode = tooltipRef.current;
  const tooltipWidth = tooltipNode.offsetWidth;
  const tooltipHeight = tooltipNode.offsetHeight;

  let tooltipX = relativeX - tooltipWidth / 2;
  let tooltipY = relativeY + 25;

  if (tooltipX + tooltipWidth > containerRect.width) {
    tooltipX = containerRect.width - tooltipWidth - 10;
  }
  if (tooltipX < 10) {
    tooltipX = 10;
  }

  if (tooltipY + tooltipHeight > containerRect.height) {
    tooltipY = relativeY - tooltipHeight - 10;
  }

  tooltipX = Math.max(10, Math.min(containerRect.width - tooltipWidth - 10, tooltipX));
  tooltipY = Math.max(10, Math.min(containerRect.height - tooltipHeight - 10, tooltipY));

  tooltip
    .style('left', `${tooltipX}px`)
    .style('top', `${tooltipY}px`);
};

export const showMergedTooltip = (
  event: MouseEvent,
  d: HierarchyPointNodeWithData,
  tooltipRef: RefObject<HTMLDivElement | null>,
  containerRef: RefObject<HTMLDivElement | null>
): void => {
  if (!tooltipRef.current || !containerRef.current || !d.data.mergedNodes) return;

  const tooltip = d3.select(tooltipRef.current);
  const containerRect = containerRef.current.getBoundingClientRect();

  const relativeX = event.clientX - containerRect.left;
  const relativeY = event.clientY - containerRect.top;

  const { tokens, probs } = d.data.mergedNodes;

  const tooltipContent = tokens.map((token, i) => `
    <div class="token-info">
      <span class="token-text">${token}</span>
      <span class="token-prob">${(probs[i] * 100).toFixed(1)}%</span>
    </div>
  `).join('');

  tooltip
    .style('opacity', 1)
    .html(`
      <div style="padding: 5px;">
        <div style="margin-bottom: 4px; font-weight: bold;">Merged Tokens:</div>
        ${tooltipContent}
      </div>
    `);

  const tooltipNode = tooltipRef.current;
  const tooltipWidth = tooltipNode.offsetWidth;
  const tooltipHeight = tooltipNode.offsetHeight;

  let tooltipX = relativeX - tooltipWidth / 2;
  let tooltipY = relativeY + 25;

  if (tooltipX + tooltipWidth > containerRect.width) {
    tooltipX = containerRect.width - tooltipWidth - 10;
  }
  if (tooltipX < 10) {
    tooltipX = 10;
  }

  if (tooltipY + tooltipHeight > containerRect.height) {
    tooltipY = relativeY - tooltipHeight - 10;
  }

  tooltipX = Math.max(10, Math.min(containerRect.width - tooltipWidth - 10, tooltipX));
  tooltipY = Math.max(10, Math.min(containerRect.height - tooltipHeight - 10, tooltipY));

  tooltip
    .style('left', `${tooltipX}px`)
    .style('top', `${tooltipY}px`);
};

export const hideTooltip = (
  tooltipRef: RefObject<HTMLDivElement | null>
): void => {
  if (!tooltipRef.current) return;
  d3.select(tooltipRef.current).style('opacity', 0);
};

export const addFoldButton = (
  parent: d3.Selection<SVGGElement, any, any, any>,
  data: HierarchyPointNodeWithData,
  onClickHandler: (event: any) => void,
  nodeHeight: number = 24
): d3.Selection<SVGGElement, any, any, any> => {
  const buttonGroup = parent.append('g');

  const nodeData = data.data;

  let allChildrenUnfolded = true;

  if (nodeData.children && nodeData.children.length > 0) {
    allChildrenUnfolded = nodeData.children.every(child =>
      child.isFolded === false || child.isFolded === undefined
    );
  }

  const shouldShowPlusIcon = !allChildrenUnfolded;

  buttonGroup.append('rect')
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
      onClickHandler(event);
    });

  buttonGroup.append('path')
    .attr('d', shouldShowPlusIcon
      ? 'M -7,0 L 7,0 M 0,-7 L 0,7'
      : 'M -7,0 L 7,0'
    )
    .attr('stroke', '#495057')
    .attr('stroke-width', 2)
    .attr('fill', 'none')
    .style('pointer-events', 'none');

  return buttonGroup;
};

export const renderPinnedIndicator = (
  nodeGroup: d3.Selection<SVGGElement, any, any, any>,
  x: number,
  y: number
): void => {
  nodeGroup.append('circle')
    .attr('cx', x)
    .attr('cy', y)
    .attr('r', 4)
    .attr('fill', '#ff9800')
    .attr('stroke', '#fff')
    .attr('stroke-width', 1);
};
