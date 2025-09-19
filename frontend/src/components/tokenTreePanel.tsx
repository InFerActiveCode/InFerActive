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
  MarkNotification
} from './tokenTreeStyles';
import {
  calculateTextWidth,
  calculateTreeDimensions,
  getSelectedNodePath,
  createHierarchyData,
  HierarchyPointNodeWithData,
  ExtendedVisualNode
} from './tokenTreeUtils';
import {
  NodeContextMenu
} from './tokenTreeInteractions';
import { findNodeById, findParentNode } from '../utils/treeTransform';
import { renderVisualization } from './tokenTreeVisualization';

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
  evaluatedNodeIds?: EvaluatedNodes;
  evaluatedPathsTotal?: number;
  evaluationFilters?: {
    showGood: boolean;
    showBad: boolean;
    showUnmarked: boolean;
  };
}

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
  evaluatedNodeIds,
  evaluatedPathsTotal,
  evaluationFilters,
}) => {
  // DOM references
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // D3 transform and zoom references
  const transformRef = useRef<d3.ZoomTransform | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  // State tracking references
  const previousRootRef = useRef<VisualNode | null>(null);

  // Explicit reset flag
  const explicitResetRef = useRef(false);

  // Initialization completion flag
  const initializedRef = useRef(false);

  // State - using default maxVisibleNodes of 10
  const [maxVisibleNodes, setMaxVisibleNodes] = useState(10);
  // Discrete node count values (1-100)
  const discreteNodeCounts = [1, 5, 10, 20, 50, 100];
  const [treeData, setTreeData] = useState<HierarchyPointNodeWithData | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [treeUpdateTrigger, setTreeUpdateTrigger] = useState(0);
  const [leftAligned, setLeftAligned] = useState(true);
  const [useSankeyLinks, setUseSankeyLinks] = useState(true);
  const [bigTokenEnabled, setBigTokenEnabled] = useState(true);
  const [useStandardTree, setUseStandardTree] = useState(false);
  const [overviewMode, setOverviewMode] = useState(false);
  
  // Remember viewport center before filtering
  const viewportCenterRef = useRef<{ x: number; y: number } | null>(null);


  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    position: { x: number, y: number };
    nodeData: any;
  }>({
    visible: false,
    position: { x: 0, y: 0 },
    nodeData: null
  });

  // Mark notification state
  const [markNotifications, setMarkNotifications] = useState<Array<{
    id: string;
    x: number;
    y: number;
    category: 'good' | 'bad';
    timestamp: number;
  }>>([]);


  // Node evaluation handler (with mark notification)
  const handleNodeEvaluate = useCallback((nodeId: string, category: 'good' | 'bad') => {
    if (!onNodeEvaluate) return;
    
    // Call original evaluation function
    onNodeEvaluate(nodeId, category);
    
    // Find node position within SVG
    if (svgRef.current && containerRef.current) {
      // Find both regular nodes and big tokens
      const nodeElement = d3.select(svgRef.current)
        .selectAll(`[data-node-id="${nodeId}"]`)
        .node() as SVGGElement;
      
      if (nodeElement) {
        const bbox = nodeElement.getBoundingClientRect();
        const containerRect = containerRef.current.getBoundingClientRect();
        
        // Calculate relative coordinates from container
        const relativeX = bbox.left - containerRect.left + bbox.width / 2;
        const relativeY = bbox.top - containerRect.top;
        
        // Add notification
        const notification = {
          id: `${nodeId}-${Date.now()}`,
          x: relativeX,
          y: relativeY,
          category,
          timestamp: Date.now()
        };
        
        setMarkNotifications(prev => [...prev, notification]);
        
        // Remove after 1 second
        setTimeout(() => {
          setMarkNotifications(prev => prev.filter(n => n.id !== notification.id));
        }, 1000);
      }
    }
  }, [onNodeEvaluate]);

  // Node generation handler
  const handleGenerateFromNode = useCallback(async (nodeId: string) => {

    if (!onGenerateFromNode) {
      console.warn('onGenerateFromNode callback not provided.');
      return null;
    }

    try {
      // Call function passed from App.tsx
      const result = await onGenerateFromNode(nodeId);
      return result;
    } catch (error) {
      console.error('error in generating:', error);
      return null;
    }
  }, [onGenerateFromNode]);

  const nodePadding = 20; // Additional spacing between nodes

  // Calculate selected token path
  const selectedNodePath = useMemo(() => {
    // If temp sequence exists and is selected
    if (tempSequence && selectedToken.sequenceId === tempSequence.id) {
      const selectedTokenInfo = tempSequence.tokens[selectedToken.tokenIndex || 0];
      if (!selectedTokenInfo) return [];
      
      return tempSequence.tokens
        .slice(0, (selectedToken.tokenIndex || 0) + 1)
        .map(t => t.nodeId);
    }
    
    // Find in existing completedSequences
    return getSelectedNodePath(completedSequences, selectedToken);
  }, [selectedToken, completedSequences, tempSequence]);

  // Tree fold/unfold handler
  const toggleFold = useCallback((nodeId: string) => {
    // Find corresponding node in original data
    if (!node) return;

    const parentNode = findNodeById(node, nodeId);
    if (!parentNode || !parentNode.children || parentNode.children.length === 0) return;

    // Check if all children are unfolded
    const allUnfolded = parentNode.children.every(child =>
      child.isFolded === false || child.isFolded === undefined
    );

    // If all children are unfolded, fold all; if any are folded, unfold all
    const newFoldState = allUnfolded;


    // Set fold state for all child nodes
    parentNode.children.forEach(child => {
      child.isFolded = newFoldState;
      child.isUserFolded = newFoldState;
    });

    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  // Calculate global Top-N nodes using frontier-based exploration
  const globalTopNNodes = useMemo(() => {
    if (!node || maxVisibleNodes > 100) return undefined;
    
    const topNodes = new Set<string>();
    const frontierSet = new Set<string>(); // Set for duplicate prevention
    const frontier: VisualNode[] = [];
    
    // Helper function to add node to frontier (duplicate prevention)
    const addToFrontier = (node: VisualNode) => {
      if (!frontierSet.has(node.id) && !topNodes.has(node.id)) {
        frontierSet.add(node.id);
        frontier.push(node);
      }
    };
    
    // Apply evaluation filter to initial frontier
    for (const child of node.children) {
      if (evaluationFilters) {
        const isGood = child.evaluationCategory === 'good' || child.ancestorEvaluation === 'good';
        const isBad = child.evaluationCategory === 'bad' || child.ancestorEvaluation === 'bad';
        const isUnmarked = !child.evaluationCategory && !child.ancestorEvaluation;
        
        const shouldHide = (isGood && !evaluationFilters.showGood) ||
                           (isBad && !evaluationFilters.showBad) ||
                           (isUnmarked && !evaluationFilters.showUnmarked);
        
        if (!shouldHide) {
          addToFrontier(child);
        }
      } else {
        addToFrontier(child);
      }
    }
    
    let count = 0;
    
    while (count < maxVisibleNodes && frontier.length > 0) {
      // Sort frontier by cumulative probability
      frontier.sort((a, b) => (b.cumulativeProb || 0) - (a.cumulativeProb || 0));
      const best = frontier.shift();
      if (!best) break;
      
      // Remove from frontier
      frontierSet.delete(best.id);
      
      // Skip if already in topNodes (already included in parent path)
      if (topNodes.has(best.id)) continue;
      
      // Greedy path: highest probability path from best to leaf
      let current = best;
      topNodes.add(current.id);
      
      // Store branch points along the path
      const pathNodes: VisualNode[] = [best];
      
      while (current.children.length > 0) {
        // Valid children with evaluation filter applied
        const validChildren = current.children.filter(child => {
          if (evaluationFilters) {
            const isGood = child.evaluationCategory === 'good' || child.ancestorEvaluation === 'good';
            const isBad = child.evaluationCategory === 'bad' || child.ancestorEvaluation === 'bad';
            const isUnmarked = !child.evaluationCategory && !child.ancestorEvaluation;
            
            const shouldHide = (isGood && !evaluationFilters.showGood) ||
                               (isBad && !evaluationFilters.showBad) ||
                               (isUnmarked && !evaluationFilters.showUnmarked);
            return !shouldHide;
          }
          return true;
        });
        
        if (validChildren.length === 0) break;
        
        // Greedy: select child with highest probability
        const greedyChild = validChildren.reduce((max, child) => 
          child.prob > max.prob ? child : max
        );
        
        topNodes.add(greedyChild.id);
        current = greedyChild;
        pathNodes.push(current);
      }
      
      // Count increases by 1 only (count entire path as 1)
      count += 1;
      
      // Add unselected siblings to frontier
      for (const pathNode of pathNodes) {
        // If this node has children
        if (pathNode.children && pathNode.children.length > 0) {
          for (const child of pathNode.children) {
            // Exclude nodes already in topNodes
            if (!topNodes.has(child.id)) {
              if (evaluationFilters) {
                const isGood = child.evaluationCategory === 'good' || child.ancestorEvaluation === 'good';
                const isBad = child.evaluationCategory === 'bad' || child.ancestorEvaluation === 'bad';
                const isUnmarked = !child.evaluationCategory && !child.ancestorEvaluation;
                
                const shouldHide = (isGood && !evaluationFilters.showGood) ||
                                   (isBad && !evaluationFilters.showBad) ||
                                   (isUnmarked && !evaluationFilters.showUnmarked);
                
                if (!shouldHide) {
                  addToFrontier(child);
                }
              } else {
                addToFrontier(child);
              }
            }
          }
        }
      }
      
      // Add best's siblings to frontier
      const parent = findParentNode(node, best.id);
      if (parent) {
        for (const sibling of parent.children) {
          if (sibling.id !== best.id && !topNodes.has(sibling.id)) {
            if (evaluationFilters) {
              const isGood = sibling.evaluationCategory === 'good' || sibling.ancestorEvaluation === 'good';
              const isBad = sibling.evaluationCategory === 'bad' || sibling.ancestorEvaluation === 'bad';
              const isUnmarked = !sibling.evaluationCategory && !sibling.ancestorEvaluation;
              
              const shouldHide = (isGood && !evaluationFilters.showGood) ||
                                 (isBad && !evaluationFilters.showBad) ||
                                 (isUnmarked && !evaluationFilters.showUnmarked);
              
              if (!shouldHide) {
                addToFrontier(sibling);
              }
            } else {
              addToFrontier(sibling);
            }
          }
        }
      }
    }
    
    return topNodes;
  }, [node, maxVisibleNodes, treeUpdateTrigger, bigTokenEnabled, evaluationFilters]);

  // Calculate tree hierarchy data
  const hierarchyData = useMemo(() => {
    return createHierarchyData(node, nodePadding, leftAligned, bigTokenEnabled, evaluationFilters, globalTopNNodes, useStandardTree, overviewMode);
  }, [node, treeUpdateTrigger, leftAligned, bigTokenEnabled, evaluationFilters, globalTopNNodes, useStandardTree, overviewMode]);

  // View reset function
  const resetViewToRoot = useCallback(() => {
    if (!svgRef.current || !containerRef.current || !zoomRef.current) return;

    const svg = d3.select(svgRef.current);
    const margin = { top: 50, right: 120, bottom: 20, left: 120 };

    // Set explicit reset flag
    explicitResetRef.current = true;

    const resetTransform = d3.zoomIdentity
      .translate(margin.left, margin.top)
      .scale(0.8);

    transformRef.current = resetTransform;


    svg.transition()
      .duration(750)
      .ease(d3.easeCubicInOut)
      .call(zoomRef.current.transform, resetTransform)
      .on('end', () => {
        explicitResetRef.current = false;
      });
  }, []);


  // Node click handler
  const handleNodeClick = useCallback((event: React.MouseEvent | MouseEvent, d: HierarchyPointNodeWithData) => {
    event.preventDefault();
    event.stopPropagation();


    if (!containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const relativeX = event.clientX - containerRect.left;
    const relativeY = event.clientY - containerRect.top;

    // Toggle context menu when clicking same node again
    if (contextMenu.visible && contextMenu.nodeData && contextMenu.nodeData.id === d.data.id) {
      setContextMenu(prev => ({ ...prev, visible: false }));
      return;
    }

    // Check if currently selected node
    const isCurrentlySelected = selectedNodePath.includes(d.data.id);

    // Deselect if clicking selected node again
    if (isCurrentlySelected && !contextMenu.visible) {
      if (onTokenSelect) {
        // Pass empty value for deselection
        onTokenSelect('', -1);
      }
      return;
    }

    // Select new node
    if (onTokenSelect) {
      // Find first sequence containing the node
      let selectedSequence = null;
      let selectedTokenIndex = -1;
      
      // Search by last node ID for big tokens
      const searchNodeId = d.data.mergedNodes?.nodeIds 
        ? d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1]
        : d.data.id;
      
      // Search all sequences by probability order
      for (const seq of completedSequences) {
        const tokenIdx = seq.tokens.findIndex(t => t.nodeId === searchNodeId);
        if (tokenIdx >= 0) {
          selectedSequence = seq;
          selectedTokenIndex = tokenIdx;
          break;
        }
      }
      
      if (selectedSequence && selectedTokenIndex >= 0) {
        // Select if found in existing sequence
        onTokenSelect(selectedSequence.id, selectedTokenIndex);
      } else {
        // Request temp sequence creation for nodes not in sequence
        // Use last node ID for big tokens
        const targetNodeId = d.data.mergedNodes?.nodeIds 
          ? d.data.mergedNodes.nodeIds[d.data.mergedNodes.nodeIds.length - 1]
          : d.data.id;
        const tempSequenceId = `temp-${targetNodeId}`;
        onTokenSelect(tempSequenceId, 0);
      }
    }

    // Show context menu
    setContextMenu({
      visible: true,
      position: { x: relativeX, y: relativeY },
      nodeData: d.data
    });
  }, [contextMenu.visible, contextMenu.nodeData, selectedNodePath, onTokenSelect, node]);

  // Close context menu
  const closeContextMenu = useCallback(() => {
    // Ignore if already closed
    if (!contextMenu.visible) return;

    setContextMenu(prev => ({ ...prev, visible: false }));
  }, [contextMenu.visible]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (!contextMenu.visible) return;

      // Check for clicks outside context menu
      const target = e.target as Element;
      const isContextMenuClick = target.closest('.context-menu-container');
      const isNodeClick = target.closest('.node') || target.closest('.node-group');

      if (!isContextMenuClick && !isNodeClick) {
        // Close menu when clicking outside menu or nodes
        closeContextMenu();
      }
    };

    // Register event listener
    document.addEventListener('mousedown', handleClickOutside);

    // Remove event listener on component unmount
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [contextMenu.visible, closeContextMenu]);

  // Link generator memoization - Sankey/Basic style selection
  const linkGenerator = useMemo(() => {
    // Function to calculate actual node width
    const getNodeWidth = (nodeData: any) => {
      // In Overview mode, filtered nodes consider only point size
      if (overviewMode && nodeData.isFiltered) {
        return 3;
      }
      
      if (nodeData.mergedNodes && nodeData.mergedNodes.tokens) {
        // For merged nodes, calculate actual rendered width
        const maxWidth = 300;
        let lineWidths = [];
        let currentLineWidth = 0;
        
        nodeData.mergedNodes.tokens.forEach((token: string, index: number) => {
          const tokenWidth = calculateTextWidth(token);
          
          if (currentLineWidth + tokenWidth > maxWidth && index > 0) {
            lineWidths.push(currentLineWidth);
            currentLineWidth = tokenWidth;
          } else {
            currentLineWidth += tokenWidth;
          }
        });
        
        if (currentLineWidth > 0) {
          lineWidths.push(currentLineWidth);
        }
        
        return Math.min(Math.max(...lineWidths, 0), maxWidth);
      } else {
        // Regular node
        return Math.min(calculateTextWidth(nodeData.token || ''), 200);
      }
    };
    
    if (useSankeyLinks) {
      // Sankey style - manual curve
      return function(d: any) {
        const sourceWidth = getNodeWidth(d.source.data);
        // Filtered nodes start from center
        const sourceX = overviewMode && d.source.data.isFiltered
          ? d.source.y
          : d.source.y + sourceWidth + 16;
        const sourceY = d.source.x;
        // Filtered nodes arrive at center
        const targetX = overviewMode && d.target.data.isFiltered 
          ? d.target.y 
          : d.target.y - 6;
        const targetY = d.target.x;
        
        const horizontalDistance = targetX - sourceX;
        const midX = sourceX + horizontalDistance * 0.4;
        return `M ${sourceX},${sourceY}
                C ${midX},${sourceY} ${midX},${targetY} ${targetX},${targetY}`;
      };
    } else {
      // Basic style - D3's natural curve
      return d3.linkHorizontal<any, any>()
        .source(d => {
          const sourceWidth = getNodeWidth(d.source.data);
          // Filtered nodes start from center
          const sourceX = overviewMode && d.source.data.isFiltered
            ? d.source.y
            : d.source.y + sourceWidth + 20;
          return [sourceX, d.source.x];
        })
        .target(d => {
          // Filtered nodes arrive at center
          const targetX = overviewMode && d.target.data.isFiltered
            ? d.target.y
            : d.target.y - 18;
          return [targetX, d.target.x];
        });
    }
  }, [useSankeyLinks, overviewMode]);

  // Color scale memoization
  const colorScale = useMemo(() => {
    return d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateBlues);
  }, []);

  // Scale for link thickness - more dramatic difference
const linkStrokeScale = useMemo(() => {
  return d3.scaleLinear()
    .domain([0, 1])
    .range([0, 20]);
}, []);





  // Check if nodes are in viewport and move to root if not
  useEffect(() => {
    if (!hierarchyData || !zoomRef.current || !transformRef.current || !svgRef.current) return;
    
    // Use setTimeout to check after rendering is complete
    const timeoutId = setTimeout(() => {
      const svg = d3.select(svgRef.current);
      const nodes = hierarchyData.descendants();
      const transform = transformRef.current;
      
      if (!transform) return;
      
      // Calculate current viewport area
      const viewportLeft = -transform.x / transform.k;
      const viewportTop = -transform.y / transform.k;
      const viewportRight = viewportLeft + dimensions.width / transform.k;
      const viewportBottom = viewportTop + dimensions.height / transform.k;
      
      // Check if there are visible nodes in viewport
      const hasVisibleNodes = nodes.some(node => {
        const nodeX = node.y;
        const nodeY = node.x;
        return nodeX >= viewportLeft - 100 && 
               nodeX <= viewportRight + 100 && 
               nodeY >= viewportTop - 50 && 
               nodeY <= viewportBottom + 50;
      });
      
      // Move to root if no visible nodes
      if (!hasVisibleNodes && nodes.length > 0 && zoomRef.current) {
        const margin = { top: 50, right: 120, bottom: 20, left: 120 };
        const resetTransform = d3.zoomIdentity
          .translate(margin.left, margin.top)
          .scale(transform.k);
        
        transformRef.current = resetTransform;
        svg.transition()
          .duration(750)
          .ease(d3.easeCubicInOut)
          .call(zoomRef.current!.transform as any, resetTransform);
      }
    }, 100);
    
    return () => clearTimeout(timeoutId);
  }, [hierarchyData, dimensions]);

  // Efficient tree data update
  useEffect(() => {
    if (!hierarchyData) return;

    // Update tree data and dimensions
    const containerWidth = containerRef.current?.clientWidth || 800;
    const containerHeight = containerRef.current?.clientHeight || 600;
    const treeDimensions = calculateTreeDimensions(
      hierarchyData,
      containerWidth,
      containerHeight
    );

    // Update only when different to prevent unnecessary re-renders
    setDimensions(prev => {
      if (prev.width !== treeDimensions.width || prev.height !== treeDimensions.height) {
        return treeDimensions;
      }
      return prev;
    });

    // Update tree data
    setTreeData(hierarchyData);
  }, [hierarchyData]);

  // Initial setup and D3 zoom settings

  // Initial setup and D3 zoom settings
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !tooltipRef.current) return;

    // Skip if already initialized
    if (initializedRef.current) return;
    initializedRef.current = true;

    const svg = d3.select(svgRef.current);

    // Initialize DOM elements only once
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
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        transformRef.current = event.transform;
        svg.select('.zoom-container').attr('transform', event.transform);
      });

    zoomRef.current = zoom;

    const margin = { top: 50, right: 120, bottom: 20, left: 120 };
    const initialTransform = d3.zoomIdentity
      .translate(margin.left, margin.top)
      .scale(0.8);

    svg.call(zoom);
    svg.call(zoom.transform, initialTransform);

    transformRef.current = initialTransform;

    // Container click event handler
    const containerClickHandler = (e: MouseEvent) => {
      // Check event target and related elements
      const targetEl = e.target as Element;

      // Check if context-menu-container element is in click path
      const contextMenuElement = targetEl.closest('.context-menu-container');
      const nodeElement = targetEl.closest('.node') || targetEl.closest('.node-group');

      // Ignore if context menu element is clicked
      if (contextMenuElement) {
        return;
      }

      // Ignore if node element is clicked
      if (nodeElement) {
        return;
      }

      // Background click - close context menu
      closeContextMenu();
    };

    // Register event listener
    containerRef.current.addEventListener('click', containerClickHandler);

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
      containerRef.current?.removeEventListener('click', containerClickHandler);
    };
  }, [closeContextMenu]);


  // User folding state reset utility
  const resetUserFolding = useCallback((node: VisualNode) => {
    node.isUserFolded = undefined;
    node.isExpanded = undefined;
    node.children.forEach(resetUserFolding);
  }, []);



  // Node count change handler
  const handleNodeCountChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    // Save current viewport center before change
    if (svgRef.current && transformRef.current) {
      const transform = transformRef.current;
      const width = dimensions.width;
      const height = dimensions.height;
      
      viewportCenterRef.current = {
        x: (width / 2 - transform.x) / transform.k,
        y: (height / 2 - transform.y) / transform.k
      };
    }
    
    const value = Number(e.target.value);
    // 101 or more means disable filtering
    setMaxVisibleNodes(value);
    
    if (node) {
      resetUserFolding(node);
    }
  }, [node, resetUserFolding, dimensions]);

  // Adjust node count by 1 with +/- buttons
  const handleDiscreteNodeCountChange = useCallback((direction: 'up' | 'down') => {
    // Save current viewport center before change
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
      // Increase by 1 (max 101)
      newValue = Math.min(maxVisibleNodes + 1, 101);
    } else {
      // Decrease by 1 (min 1)
      newValue = Math.max(maxVisibleNodes - 1, 1);
    }
    
    if (newValue === maxVisibleNodes) return;
    
    setMaxVisibleNodes(newValue);
    
    if (node) {
      resetUserFolding(node);
    }
  }, [maxVisibleNodes, node, resetUserFolding, dimensions]);

  const handleSeparateBigToken = useCallback((nodeData: any, clickedTokenIndex?: number) => {
  
    if (!node) return;
    if (!nodeData.mergedNodes) console.warn(`no big token`);
    // Use nodeData directly
    if (nodeData.mergedNodes && nodeData.mergedNodes.nodeIds) {

      if (clickedTokenIndex !== undefined) {
        // Expand only clicked token
        const clickedNodeId = nodeData.mergedNodes.nodeIds[clickedTokenIndex];
        const originalNode = findNodeById(node, clickedNodeId);
        if (originalNode) {
          originalNode.isExpanded = true;
        }
      } else {
        // Existing behavior: expand all nodes
        for (const id of nodeData.mergedNodes.nodeIds) {
          const originalNode = findNodeById(node, id);
          if (originalNode) {
            originalNode.isExpanded = true;
          }
        }
      }

      // Trigger tree update
      setTreeUpdateTrigger(prev => prev + 1);
    } else {
      console.warn(`no big token data in nodeData`);
    }
  }, [node]);

  const handleMergeToBigToken = useCallback((nodeData: any) => {

    if (!node) return;

    // Find current node in original tree
    const currentNode = findNodeById(node, nodeData.id);
    if (!currentNode) return;

    // Reset expansion state of current node only
    if (currentNode.isExpanded) {
      currentNode.isExpanded = false;
    }

    // Trigger tree update
    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  const handleFoldNode = useCallback((nodeId: string) => {

    if (!node) return;

    // Find node in original tree
    const targetNode = findNodeById(node, nodeId);
    if (!targetNode) return;

    // Set isUserFolded to fold node
    targetNode.isUserFolded = true;
    targetNode.isFolded = true;

    // Trigger tree update
    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  // Filtered node click handler - opposite of handleFoldNode logic
  const handleFilteredNodeClick = useCallback((nodeId: string) => {

    if (!node) return;

    // Find node in original tree
    const targetNode = findNodeById(node, nodeId);
    if (!targetNode) return;

    // Set isUserFolded to false to unfold node
    targetNode.isUserFolded = false;
    targetNode.isFolded = false;

    // Trigger tree update
    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  // Function to unfold adjacent folded sibling nodes
  const handleUnfoldAdjacentSiblings = useCallback((nodeId: string) => {

    if (!node) return;

    // Find current node
    const currentNode = findNodeById(node, nodeId);
    if (!currentNode) return;

    // Find parent node
    const parentNode = findParentNode(node, nodeId);
    if (!parentNode || !parentNode.children) return;

    // Find current node index
    const currentIndex = parentNode.children.findIndex(child => child.id === nodeId);
    if (currentIndex === -1) return;

    // Unfold consecutive folded siblings immediately following
    for (let i = currentIndex + 1; i < parentNode.children.length; i++) {
      const sibling = parentNode.children[i];
      if (sibling.isFolded || sibling.isUserFolded) {
        sibling.isUserFolded = false;
        sibling.isFolded = false;
      } else {
        break; // Stop when unfolded node is encountered
      }
    }

    // Trigger tree update
    setTreeUpdateTrigger(prev => prev + 1);
  }, [node]);

  // Visualization rendering callback
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
      configSettings: { contextMenuEnabled: true },
      overviewMode,
      bigTokenEnabled
    });
  }, [dimensions, hierarchyData, selectedNodePath, node, toggleFold,
    handleNodeClick, linkGenerator, colorScale, linkStrokeScale,
    useSankeyLinks, handleGenerateFromNode, onTokenSelect,
    handleSeparateBigToken, handleMergeToBigToken, handleFoldNode, handleUnfoldAdjacentSiblings, handleFilteredNodeClick, contextMenu, closeContextMenu, overviewMode, bigTokenEnabled]);

  // Deselect if selected path is not fully rendered
  useEffect(() => {
    if (!hierarchyData || selectedNodePath.length === 0) {
      return;
    }

    const renderedNodeIds = new Set(
      hierarchyData.descendants().map(node => node.data.id)
    );

    const isFullPathRendered = selectedNodePath.every(id => renderedNodeIds.has(id));

    if (!isFullPathRendered) {
      if (onTokenSelect) {
        onTokenSelect('', -1);
      }
    }
  }, [hierarchyData]); // Check only on hierarchyData changes

  // D3 rendering
  useEffect(() => {
    renderVisualizationCallback();
  }, [treeData, selectedNodePath, dimensions, renderVisualizationCallback]);

  return (
    <TreeContainer ref={containerRef}>
      {true && (
        <ToggleControl>

          <SankeyButton
            isActive={bigTokenEnabled}
            onClick={() => setBigTokenEnabled(!bigTokenEnabled)}
            title={bigTokenEnabled ? "Disable Big Tokens" : "Enable Big Tokens"}
          >
            {bigTokenEnabled ? 'M' : 'I'}
          </SankeyButton>

          <SankeyButton
            isActive={overviewMode}
            onClick={() => setOverviewMode(!overviewMode)}
            title={overviewMode ? "Detailed View" : "Overview Mode"}
          >
            O
          </SankeyButton>
        </ToggleControl>
      )}
      
      {false && (
        <ZoomControl style={{ top: '16px', width: '56px' }}>
          <SankeyButton
            isActive={overviewMode}
            onClick={() => setOverviewMode(!overviewMode)}
            title={overviewMode ? "Detailed View" : "Overview Mode"}
            style={{ width: '42px', height: '42px' }}
          >
            O
          </SankeyButton>
        </ZoomControl>
      )}

      <ZoomControl>
        <HomeButton onClick={resetViewToRoot} title="Go to Root">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#333" strokeWidth="2">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
          </svg>
        </HomeButton>

        <VerticalSliderContainer>
          <SliderText style={{ fontSize: '11px' }}>
            {maxVisibleNodes > 100 ? 'All' : `Top ${maxVisibleNodes}`}
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
              min="1"
              max="101"
              value={Math.min(maxVisibleNodes, 101)}
              onChange={handleNodeCountChange}
              step="1"
            />
            <SliderTicks>
              {[...discreteNodeCounts, 101].map((count, index) => {
                const position = ((count - 1) / 100) * 100;
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
          // Close context menu when clicking empty space in SVG
          const targetEl = e.target as Element;

          // If clicked element is SVG itself or zoom-container (not a node)
          if (targetEl.tagName === 'svg' || targetEl.classList.contains('zoom-container')) {
            closeContextMenu();
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