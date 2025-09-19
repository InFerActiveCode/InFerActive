import * as d3 from 'd3';
import { findNodeById, findParentNode } from '../utils/treeTransform';
import { 
  calculateTextWidth, 
  calculateTextWrapping,
  HierarchyPointNodeWithData 
} from './tokenTreeUtils';
import {
  showTooltip,
  showMergedTooltip,
  hideTooltip,
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
  handleGenerateFromNode: (nodeId: string) => Promise<any>;
  toggleFold: (nodeId: string) => void;
  onTokenSelect?: (sequenceId: string, tokenIndex: number) => void;
  handleSeparateBigToken: (nodeData: any, clickedTokenIndex?: number) => void;
  handleMergeToBigToken: (nodeData: any) => void;
  handleFoldNode: (nodeId: string) => void;
  handleUnfoldAdjacentSiblings: (nodeId: string) => void;
  handleFilteredNodeClick?: (nodeId: string) => void;
  contextMenu: {
    visible: boolean;
    position: { x: number, y: number };
    nodeData: any;
  };
  closeContextMenu: () => void;
  configSettings?: {
    contextMenuEnabled: boolean;
  };
  unfoldPathToNode?: (nodeId: string) => void;
  overviewMode?: boolean;
  bigTokenEnabled?: boolean;
}

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
    handleGenerateFromNode,
    toggleFold,
    onTokenSelect,
    handleSeparateBigToken,
    handleMergeToBigToken,
    handleFoldNode,
    handleUnfoldAdjacentSiblings,
    handleFilteredNodeClick,
    contextMenu,
    closeContextMenu,
    configSettings,
    overviewMode = false,
    bigTokenEnabled = true
  } = params;

  if (!svgRef.current || !containerRef.current || !hierarchyData) return;

  const svg = d3.select(svgRef.current);
  const g = svg.select<SVGGElement>('.zoom-container');

  svg
    .attr('width', dimensions.width)
    .attr('height', dimensions.height);

  const shouldResetView = (() => {
    return explicitResetRef.current || !previousRootRef.current;
  })();

  previousRootRef.current = node;

  if (shouldResetView && zoomRef.current) {
    const margin = { top: 50, right: 120, bottom: 20, left: 120 };
    const resetTransform = d3.zoomIdentity
      .translate(margin.left, margin.top)
      .scale(0.8);

    transformRef.current = resetTransform;

    svg.transition()
      .duration(750)
      .call(zoomRef.current.transform, resetTransform);
  } else if (transformRef.current) {
    if (viewportCenterRef.current && zoomRef.current) {
      const currentScale = transformRef.current.k;
      const newX = dimensions.width / 2 - viewportCenterRef.current.x * currentScale;
      const newY = dimensions.height / 2 - viewportCenterRef.current.y * currentScale;
      
      const newTransform = d3.zoomIdentity
        .translate(newX, newY)
        .scale(currentScale);
      
      transformRef.current = newTransform;
      svg.call(zoomRef.current.transform, newTransform);

      viewportCenterRef.current = null;
    } else {
      g.attr('transform', transformRef.current.toString());
    }
  }

  const links = hierarchyData.links();
  const linkSelection = g.select('.links')
    .selectAll<SVGPathElement, any>('path')
    .data(links, (d: any) => `${d.source.data.id}-${d.target.data.id}`); 

  linkSelection.exit().remove();

  const linkEnter = linkSelection.enter()
    .append('path')
    .attr('class', 'link')
    .attr('fill', 'none')
    .attr('stroke-linecap', 'butt') 
    .attr('opacity', 1) 
    .attr('d', linkGenerator); 

  
  const allLinks = linkEnter.merge(linkSelection);
  
  
  allLinks
    .attr('stroke', d => {
      const sourceId = d.source.data.id;
      const targetId = d.target.data.id;
      const targetNode = d.target.data;
      
      
      if (selectedNodePath.includes(sourceId) && selectedNodePath.includes(targetId)) {
        return '#2196f3';
      }
      const evaluationStatus = targetNode.evaluationCategory || targetNode.ancestorEvaluation;
      
      if (evaluationStatus === 'good') {
        if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
          return '#81c784'; 
        }
        return '#4caf50'; 
      } else if (evaluationStatus === 'bad') {
        if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
          return '#e57373'; 
        }
        return '#f44336'; 
      }
      
      if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
        return '#cccccc';
      }
      
      return '#87CEEB';
    })
    .attr('stroke-width', d => {
      if (useSankeyLinks) {
        const baseWidth = linkStrokeScale(d.target.data.prob);
        if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
          return baseWidth * 0.25;
        }
        return baseWidth;
      } else {
        return 2;
      }
    })
    .attr('opacity', d => {
      if (overviewMode && (d.source.data.isFiltered || d.target.data.isFiltered)) {
        return 0.7;
      }
      return 1;
    });
  
  allLinks
    .transition()
    .duration(300)
    .attr('d', linkGenerator);

  const nodesToRender = hierarchyData.descendants();  
  
    const nodeSelection = g.select('.nodes')
      .selectAll<SVGGElement, any>('.node')
      .data(nodesToRender, (d: any) => d.data.id); 

    nodeSelection.exit().remove();

    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'node')
      .attr('data-node-id', d => d.data.id)
      .attr('transform', d => `translate(${d.y},${d.x})`);

    nodeSelection
      .attr('data-node-id', d => d.data.id)
      .transition()
      .duration(300)
      .attr('transform', d => `translate(${d.y},${d.x})`);

    const allNodes = nodeEnter.merge(nodeSelection);
    
    allNodes.each(function (d) {
    const container = d3.select(this);

    
    container.selectAll('*').remove();

    
    if (overviewMode && d.data.isFiltered) {
      
      const evaluationStatus = d.data.evaluationCategory || d.data.ancestorEvaluation;
      
      let fillColor = '#cccccc'; 
      let strokeColor = '#999999';
      
      if (evaluationStatus === 'good') {
        fillColor = '#81c784'; 
        strokeColor = '#4caf50';
      } else if (evaluationStatus === 'bad') {
        fillColor = '#e57373'; 
        strokeColor = '#f44336'; 
      }
      
      const hitArea = container.append('circle')
        .attr('r', 12) 
        .attr('fill', 'transparent')
        .attr('stroke', 'none')
        .style('cursor', 'pointer');
      
  
      const filteredNodeCircle = container.append('circle')
        .attr('r', 3)
        .attr('fill', fillColor)
        .attr('stroke', strokeColor)
        .attr('stroke-width', 1)
        .attr('opacity', 0.7)
        .style('pointer-events', 'none'); 

      if (handleFilteredNodeClick) {
        hitArea.on('click', function(event: MouseEvent) {
          event.stopPropagation();
          handleFilteredNodeClick(d.data.id);
        });
        
        hitArea
          .on('mouseover', function() {
            filteredNodeCircle
              .attr('r', 8)
              .attr('stroke-width', 2);
          })
          .on('mouseout', function() {
            filteredNodeCircle
              .attr('r', 3)
              .attr('stroke-width', 1);
          });
      }
      
      return; 
    }

    const nodeContainer = container.append('g')
      .attr('class', 'node-container')
      .attr('data-node-id', d.data.mergedNodes && d.data.mergedNodes.nodeIds 
        ? d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1]
        : d.data.id);

    let actualWidth = 0;
    let actualHeight = 28;
    
    if (d.data.mergedNodes && d.data.mergedNodes.tokens) {
      const maxWidth = 300;
      let lineWidths = [];
      let currentLineWidth = 0;
      let totalLines = 1;
      
      d.data.mergedNodes.tokens.forEach((token: string, index: number) => {
        const tokenWidth = calculateTextWidth(token);
        
        if (currentLineWidth + tokenWidth > maxWidth && index > 0) {
          lineWidths.push(currentLineWidth);
          currentLineWidth = tokenWidth;
          totalLines++;
        } else {
          currentLineWidth += tokenWidth;
        }
      });
      
      if (currentLineWidth > 0) {
        lineWidths.push(currentLineWidth);
      }
      
      actualWidth = Math.max(...lineWidths, 0);
      actualHeight = totalLines * 16 + 12; 
    } else {
      
      actualWidth = calculateTextWidth(d.data.token || '');
      actualHeight = 28;
    }
    
    const textWidth = actualWidth;
    const textWrapping = { lines: [], height: actualHeight };
    const originalNode = findNodeById(node, d.data.id);

    let hasChildren = false;
    let mergedLastNodeId: string | null = null;

    if (d.data.mergedNodes && d.data.mergedNodes.nodeIds && d.data.mergedNodes.nodeIds.length > 0) {
      
      mergedLastNodeId = d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1];
      
      const lastNode = findNodeById(node, mergedLastNodeId);
      hasChildren = lastNode ? (lastNode.children && lastNode.children.length > 0) : false;
    } else {
      
      hasChildren = originalNode ? (originalNode.children && originalNode.children.length > 0) : false;
    }

   
    const maxNodeWidth = d.data.mergedNodes ? 300 : 200;
    const totalWidth = Math.min(textWidth, maxNodeWidth) + 20; 
    const nodeHeight = textWrapping.height;
    const isSelected = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        
        return d.data.mergedNodes.nodeIds.some(nodeId => 
          selectedNodePath.includes(nodeId)
        );
      }
      
      return selectedNodePath.includes(d.data.id);
    })();
    const isPinned = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
       
        return d.data.mergedNodes.nodeIds.some(nodeId => {
          const targetNode = findNodeById(node, nodeId);
          return targetNode?.isPinned || false;
        });
      } else {
        
        return Boolean(d.data.isPinned);
      }
    })();

    const isEvaluated = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
    
        return d.data.mergedNodes.nodeIds.some(nodeId => {
          const targetNode = findNodeById(node, nodeId);
          return targetNode?.isEvaluated || false;
        });
      } else {
        
        return Boolean(d.data.isEvaluated);
      }
    })();

    const evaluationCategory = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
 
        for (const nodeId of d.data.mergedNodes.nodeIds) {
          const targetNode = findNodeById(node, nodeId);
          if (targetNode?.evaluationCategory) {
            return targetNode.evaluationCategory;
          }
        }
        return null;
      } else {
        
        return d.data.evaluationCategory || null;
      }
    })();

    const ancestorEvaluation = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {

        const hasDirectEvaluation = d.data.mergedNodes.nodeIds.some(nodeId => {
          const targetNode = findNodeById(node, nodeId);
          return targetNode?.evaluationCategory;
        });
        
        if (hasDirectEvaluation) {
          return null;
        }
        
        for (const nodeId of d.data.mergedNodes.nodeIds) {
          const targetNode = findNodeById(node, nodeId);
          if (targetNode?.ancestorEvaluation) {
            return targetNode.ancestorEvaluation;
          }
        }
        return null;
      } else {
        
        return d.data.ancestorEvaluation || null;
      }
    })();


    const isGenerating = (() => {
      if (d.data.mergedNodes && d.data.mergedNodes.nodeIds) {
        
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

   
    if (d.data.foldedSiblingCount && d.data.foldedSiblingCount > 0) {
      
      const stackGroup = nodeGroup.append('g')
        .attr('class', 'stack-group');
      
      if (d.data.foldedSiblingCount >= 2) {
        stackGroup.append('rect')
          .attr('x', -6 + 4)  
          .attr('y', -nodeHeight/2 + 10)
          .attr('width', totalWidth + 12 - 8) 
          .attr('height', nodeHeight)
          .attr('fill', '#f8f9fa')
          .attr('stroke', '#adb5bd')
          .attr('stroke-width', 1)
          .attr('rx', 6);
      }
      
      
      stackGroup.append('rect')
        .attr('x', -6 + 2) 
        .attr('y', -nodeHeight/2 + 5)
        .attr('width', totalWidth + 12 - 4)
        .attr('height', nodeHeight)
        .attr('fill', '#f8f9fa')
        .attr('stroke', '#ced4da')
        .attr('stroke-width', 1)
        .attr('rx', 6);
      
      nodeGroup.insert('rect', ':first-child')  
        .attr('class', 'stack-hover-area')
        .attr('x', -10)
        .attr('y', -nodeHeight/2 + 3)  
        .attr('width', totalWidth + 20)
        .attr('height', nodeHeight + 25)  
        .attr('fill', 'transparent')
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          event.stopPropagation();
          stackGroup.selectAll('rect')
            .attr('fill', '#e9ecef')
            .attr('stroke', '#6c757d');
          if (tooltipRef.current && containerRef.current) {
            const tooltip = d3.select(tooltipRef.current);
            const containerRect = containerRef.current.getBoundingClientRect();
            
            tooltip
              .style('opacity', 1)
              .style('left', `${event.clientX - containerRect.left}px`)
              .style('top', `${event.clientY - containerRect.top + 20}px`)
              .html('<div style="padding: 5px;">Click to show hidden siblings</div>');
          }
        })
        .on('mouseout', function(event) {
          event.stopPropagation();
          stackGroup.select('rect:first-child')
            .attr('fill', '#f8f9fa')
            .attr('stroke', '#adb5bd');
          stackGroup.select('rect:last-child')
            .attr('fill', '#f8f9fa')
            .attr('stroke', '#ced4da');
          hideTooltip(tooltipRef);
        })
        .on('click', function(event) {
          event.stopPropagation();
          handleUnfoldAdjacentSiblings(d.data.id);
        });
    }

    const background = nodeGroup.append('rect')
      .attr('x', -6)
      .attr('y', -nodeHeight/2)
      .attr('width', totalWidth + 12)
      .attr('height', nodeHeight)
      .attr('fill', (() => {
        if (isSelected) return '#e3f2fd'; 
        if (isPinned) return '#fff8e1';  
        return 'white'; 
      })())
      .attr('stroke', (() => {
        
        if (evaluationCategory === 'good' || ancestorEvaluation === 'good') return '#4caf50';
        if (evaluationCategory === 'bad' || ancestorEvaluation === 'bad') return '#f44336';  
        if (isSelected) return '#2196f3';
        if (isPinned) return '#ffb74d'; 
        return '#dee2e6'; 
      })())
      .attr('stroke-width', (evaluationCategory || ancestorEvaluation) ? 3 : (isSelected || isPinned ? 2 : 1))
      .attr('rx', 6)
      .style('opacity', 1)
      .style('cursor', 'pointer');

    
    const handleMouseOver = function (event: any) {
      background.attr('stroke', 
        (evaluationCategory === 'good' || ancestorEvaluation === 'good') ? '#388e3c' :  
        (evaluationCategory === 'bad' || ancestorEvaluation === 'bad') ? '#c62828' :    
        isSelected ? '#64b5f6' : (isPinned ? '#ffa726' : '#adb5bd'));
      
      if (!d.data.mergedNodes) {
        showTooltip(event, d, tooltipRef, containerRef);
      }
    };

    const handleMouseOut = function () {
      background.attr('stroke', 
        (evaluationCategory === 'good' || ancestorEvaluation === 'good') ? '#4caf50' :  
        (evaluationCategory === 'bad' || ancestorEvaluation === 'bad') ? '#f44336' :    
        isSelected ? '#90caf9' : (isPinned ? '#ffb74d' : '#dee2e6'));
      hideTooltip(tooltipRef);
    };

    background
      .on('mouseover', handleMouseOver)
      .on('mouseout', handleMouseOut)
      .on('click', (event) => handleNodeClick(event, d));

    nodeGroup.select('text')
      .on('mouseover', handleMouseOver)
      .on('mouseout', handleMouseOut)
      .on('click', (event) => handleNodeClick(event, d));

    if (d.data.mergedNodes) {
      background
        .attr('stroke-width', '2')
    }

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
      .attr('font-weight', isSelected ? 'bold' : 'normal');
    
    if (d.data.mergedNodes && d.data.mergedNodes.tokens) {
      let currentLineWidth = 0;
      let currentLineIndex = 0;
      let isNewLine = true;
      const maxWidth = 300; 

      let totalLines = 1;
      let tempWidth = 0;
      d.data.mergedNodes.tokens.forEach((token: string) => {
        const tokenWidth = calculateTextWidth(token);
        if (tempWidth + tokenWidth > maxWidth && tempWidth > 0) {
          totalLines++;
          tempWidth = tokenWidth;
        } else {
          tempWidth += tokenWidth;
        }
      });
      
      textElement.attr('y', -(totalLines - 1) * 8);
      
      d.data.mergedNodes.tokens.forEach((token: string, tokenIndex: number) => {
        const tokenWidth = calculateTextWidth(token);
        
        if (currentLineWidth + tokenWidth > maxWidth && tokenIndex > 0) {
          currentLineIndex++;
          currentLineWidth = 0;
          isNewLine = true;
        }
        
        const tokenSpan = textElement.append('tspan')
          .text(token)
          .style('cursor', 'pointer')
          .attr('class', 'merged-token')
          .attr('data-token-index', tokenIndex);
        
        if (isNewLine) {
          tokenSpan.attr('x', 0)
            .attr('dy', currentLineIndex === 0 ? '0.31em' : '16px');
          isNewLine = false;
        }
        
        currentLineWidth += tokenWidth;
        
        tokenSpan
          .on('mouseover', function(event) {
            d3.select(this)
              .attr('fill', '#2196f3')
              .style('text-decoration', 'underline');
            
            if (tooltipRef.current && containerRef.current) {
              const tooltip = d3.select(tooltipRef.current);
              const containerRect = containerRef.current.getBoundingClientRect();
              
              const tokenProb = d.data.mergedNodes?.probs?.[tokenIndex] || 0;
       
              const tokenNodeId = d.data.mergedNodes?.nodeIds?.[tokenIndex];
              let tokenCumulativeProb = d.data.cumulativeProb || d.data.prob;
              
              if (tokenIndex > 0 && d.data.mergedNodes?.probs) {
                for (let i = 1; i <= tokenIndex; i++) {
                  tokenCumulativeProb *= d.data.mergedNodes.probs[i];
                }
              }
              
              tooltip
                .style('opacity', 1)
                .style('left', `${event.clientX - containerRect.left}px`)
                .style('top', `${event.clientY - containerRect.top + 20}px`)
                .html(`
                  <div style="padding: 5px;">
                    <div><strong>${token}</strong></div>
                    <div>Probability: ${(tokenProb * 100).toFixed(1)}%</div>
                    <div>Cumulative: ${(tokenCumulativeProb * 100).toFixed(2)}%</div>
                  </div>
                `);
            }
          })
          .on('mouseout', function() {
            d3.select(this)
              .attr('fill', null)
              .style('text-decoration', null);
            
            hideTooltip(tooltipRef);
          })
          .on('click', function(event) {
            event.stopPropagation();
            
            if (!contextMenu.visible || contextMenu.nodeData?.id !== d.data.id) {
              handleNodeClick(event, d);
              return;
            }
            
            const targetNodeId = d.data.mergedNodes?.nodeIds?.[tokenIndex];
            handleSeparateBigToken(d.data, tokenIndex);
            
            if (targetNodeId && onTokenSelect) {
              setTimeout(() => {
                const tempSequenceId = `temp-${targetNodeId}`;
                onTokenSelect(tempSequenceId, 0);
              }, 10);
            }
            
            closeContextMenu();
            
            hideTooltip(tooltipRef);
          });
      });
    } else {
      
      textElement.append('tspan')
        .attr('dy', '0.31em')
        .text(d.data.token || '');
    }

    if (isPinned) {
      renderPinnedIndicator(nodeGroup, -2, -8);
    }

    nodeGroup
      .on('mouseover', handleMouseOver)
      .on('mouseout', handleMouseOut)
      .on('click', function (event) {
        handleNodeClick(event, d);
      });
    const hasSiblings = d.parent && d.parent.children && d.parent.children.length > 1;
    
    {
      const leftButtonGroup = nodeContainer.append('g')
        .attr('class', 'left-button-group')
        .attr('transform', `translate(-15, 0)`)
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
          if (!hasSiblings && bigTokenEnabled) {
            if (d.data.isExpanded) {
              handleMergeToBigToken(d.data);
            } else {
              handleSeparateBigToken(d.data, 0);
            }
          } else {
            handleFoldNode(d.data.id);
          }
        });

    if (!hasSiblings && bigTokenEnabled) {
      leftButtonGroup.append('path')
          .attr('d', 'M 2,-4 L -2,0 L 2,4 M 6,-4 L 2,0 L 6,4')
          .attr('stroke', '#495057')
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .style('pointer-events', 'none');
    } else {
      leftButtonGroup.append('path')
          .attr('d', 'M -7,0 L 7,0')
          .attr('stroke', '#495057')
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .style('pointer-events', 'none');
    }

    const leftHoverArea = nodeContainer.append('rect')
      .attr('class', 'left-hover-area')
      .attr('x', -30)
      .attr('y', -nodeHeight/2)
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
        if (!hasSiblings && bigTokenEnabled) {
          if (d.data.isExpanded) {
            handleMergeToBigToken(d.data);
          } else {
            handleSeparateBigToken(d.data, 0);
          }
        } else {
          handleFoldNode(d.data.id);
        }
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
    
    if (hasChildren || !hasChildren) {
      const buttonGroup = nodeContainer.append('g')
        .attr('class', hasChildren ? 'fold-button-group' : 'generate-button-group')
        .attr('transform', `translate(${textWidth + 15}, 0)`)
        .style('opacity', 0);

      if (hasChildren) {
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
      } else {
        
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
              .attr('fill', '#e3f2fd')
              .attr('stroke', '#64b5f6');
          })
          .on('mouseout', function() {
            d3.select(this)
              .attr('fill', 'white')
              .attr('stroke', '#dee2e6');
          })
          .on('click', function(event) {
            event.stopPropagation();
            
            
            const targetNodeId = mergedLastNodeId || d.data.id;
            handleGenerateFromNode(targetNodeId);
          });

        buttonGroup.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '0.3em')
          .attr('font-size', '12px')
          .attr('fill', '#2196f3')
          .style('pointer-events', 'none')
          .text('▶');
      }

      const buttonHoverArea = nodeContainer.append('rect')
        .attr('class', 'button-hover-area')
        .attr('x', textWidth + 6)
        .attr('y', -nodeHeight/2)
        .attr('width', 28)
        .attr('height', nodeHeight)
        .attr('fill', 'transparent')
        .style('cursor', 'pointer');

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
          
          if (hasChildren) {
            if (mergedLastNodeId) {
              toggleFold(mergedLastNodeId);
            } else {
              toggleFold(d.data.id);
            }
          } else {
            const targetNodeId = mergedLastNodeId || d.data.id;
            handleGenerateFromNode(targetNodeId);
          }
        });

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
  });

  explicitResetRef.current = false;

};