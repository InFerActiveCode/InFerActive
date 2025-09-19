import * as d3 from 'd3';
import { flextree } from 'd3-flextree';
import leftAlignedFlextree from '../utils/leftAlignedFlextree';
import { VisualNode, CompletedSequence, TokenSelection } from '../types/types';
import { findNodeById } from '../utils/treeTransform';


export interface ExtendedVisualNode extends VisualNode {
  isPinned?: boolean;
  isGenerating?: boolean;
  isCompleted?: boolean;
}


export type HierarchyPointNodeWithData = d3.HierarchyPointNode<ExtendedVisualNode> & {
  nodeWidth?: number;
};


export const calculateTextWidth = (text: string): number => {
  
  const minWidth = 5;
  
  if (!text) return minWidth;
  
  
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (context) {
    
    context.font = '12px monospace';
    const metrics = context.measureText(text);
    let width = Math.ceil(metrics.width);
    
    
    const isWindows = navigator.userAgent.toLowerCase().includes('win');
    if (isWindows) {
      width = Math.ceil(width * 1.15);
    }
    
    return Math.max(minWidth, width);
  }
  
  
  return Math.max(minWidth, text.length * 7);
};


export const calculateTextWrapping = (text: string, maxWidth: number = 500): { lines: string[], height: number } => {
  if (!text) return { lines: [''], height: 28 };
  
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) {
    return { lines: [text], height: 28 };
  }
  
  context.font = '12px monospace';
  const words = text.split(' ');
  const lines: string[] = [];
  let currentLine = '';
  
  for (const word of words) {
    const testLine = currentLine ? currentLine + ' ' + word : word;
    const metrics = context.measureText(testLine);
    
    if (metrics.width > maxWidth && currentLine) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = testLine;
    }
  }
  
  if (currentLine) {
    lines.push(currentLine);
  }
  
  const lineHeight = 16;
  const padding = 12;
  const height = lines.length * lineHeight + padding;
  
  return { lines, height };
};


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


export const getVisibleChildren = (
  node: VisualNode,
  bigTokenEnabled: boolean = true,
  globalTopNNodes?: Set<string>
): VisualNode[] => {
  
  if (node.isFolded) {
    return [];
  }

  let children = [...node.children];
  
  
  const greedyChild = children.length > 0 ?
    children.reduce((max, child) => child.prob > max.prob ? child : max) : null;
  
  
  if (globalTopNNodes) {
    
    for (const child of children) {
      const isInTopN = globalTopNNodes.has(child.id);
      
      
      if (node.isUserFolded == false && greedyChild && child.id === greedyChild.id) {
        child.isUserFolded = false;
      }
      
      
      if (child.isUserFolded !== undefined) {
        child.isFolded = child.isUserFolded;
      } else {
        child.isFolded = !isInTopN;
      }
    }
  } else {
    
    for (const child of children) {
      
      if (child.isUserFolded !== undefined) {
        child.isFolded = child.isUserFolded;
      } else {
        
        if (greedyChild && child.id === greedyChild.id) {
          child.isFolded = false;
        } else {
          child.isFolded = false;
        }
      }
    }
  }
  
  
  const visibleChildren = children.filter(child => !child.isFolded);
  
  
  const processedChildren = visibleChildren.map((child) => {
    
    const childIndex = children.indexOf(child);
    let foldedAfter = 0;
    for (let i = childIndex + 1; i < children.length; i++) {
      if (children[i].isFolded) {
        foldedAfter++;
      } else {
        break; 
      }
    }
    
    
    const childWithFoldedInfo = {
      ...child,
      foldedSiblingCount: foldedAfter
    };
    
    let current = childWithFoldedInfo;
    const mergedTokens = [current.token];
    const mergedProbs = [current.prob];
    const mergedNodeIds = [current.id];
    
    
    let nextChildren = current.children || [];
    
    
    for (const nextChild of nextChildren) {
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
    
    
    nextChildren = nextChildren.filter(nextChild => !nextChild.isFolded);
    
    
    while (nextChildren.length === 1 && !nextChildren[0].isExpanded) {
      current = { ...nextChildren[0], foldedSiblingCount: foldedAfter };
      mergedTokens.push(current.token);
      mergedProbs.push(current.prob);
      mergedNodeIds.push(current.id);
      
      
      nextChildren = current.children || [];
      
      
      for (const nextChild of nextChildren) {
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
      
      
      nextChildren = nextChildren.filter(nextChild => !nextChild.isFolded);
    }
    
    
    if (bigTokenEnabled && mergedTokens.length > 1) {
      return {
        ...childWithFoldedInfo,
        token: mergedTokens.join(''),
        mergedNodes: {
          tokens: mergedTokens,
          probs: mergedProbs,
          nodeIds: mergedNodeIds
        },
        children: current.children
      };
    }
    
    return childWithFoldedInfo;
  });

  return processedChildren;
};


export const getVisibleChildrenForOverview = (
  node: VisualNode,
  bigTokenEnabled: boolean = true,
  globalTopNNodes?: Set<string>
): VisualNode[] => {
  
  const children = [...node.children];
  
  
  const greedyChild = children.length > 0 ? 
    children.reduce((max, child) => child.prob > max.prob ? child : max) : null;
  
  
  const processedChildren = children.map((child) => {
    
    const isFiltered = globalTopNNodes ? !globalTopNNodes.has(child.id) : false;
    

    
    
    let shouldFilter = isFiltered
    if (node.isUserFolded === false) {
      
      if (child.isUserFolded != true && greedyChild && child.id === greedyChild.id) {
        child.isUserFolded = false;
      }
    }
    
    if (node.isFiltered === false && child.isUserFolded !== undefined) {
      shouldFilter = child.isUserFolded;
    }
    else {
      shouldFilter = isFiltered;
    }
    
    
    let current = child;
    const mergedTokens = [current.token];
    const mergedProbs = [current.prob];
    const mergedNodeIds = [current.id];
    
    
    let nextChildren = current.children || [];
    
    
    
    while (nextChildren.length === 1 && !nextChildren[0].isExpanded) {
      current = nextChildren[0];
      mergedTokens.push(current.token);
      mergedProbs.push(current.prob);
      mergedNodeIds.push(current.id);
      
      
      nextChildren = current.children || [];
    }
    
    
    if (bigTokenEnabled && mergedTokens.length > 1) {
      return {
        ...child,
        token: mergedTokens.join(''),
        mergedNodes: {
          tokens: mergedTokens,
          probs: mergedProbs,
          nodeIds: mergedNodeIds
        },
        children: current.children,
        isFiltered: shouldFilter,
        isFolded: false
      };
    }
    
    return {
      ...child,
      isFiltered: shouldFilter,
      isFolded: false
    };
  });
  
  return processedChildren;
};


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
  globalTopNNodes?: Set<string>,
  useStandardTree: boolean = false,
  overviewMode: boolean = false
): HierarchyPointNodeWithData | null => {
  if (!node) return null;
  
  const getChildrenFunction = overviewMode
    ? (n: VisualNode) => getVisibleChildrenForOverview(n, bigTokenEnabled, globalTopNNodes)
    : (n: VisualNode) => getVisibleChildren(n, bigTokenEnabled, globalTopNNodes);
  
  
  const hierarchy = d3.hierarchy(node, getChildrenFunction);
  
  
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
          
          if (overviewMode && n.data.isFiltered) {
            return [-1, 50] as [number, number];
          }
          
          const textWrapping = calculateTextWrapping(n.data.token || '');
          const textWidth = calculateTextWidth(n.data.token || '');
          const hasChildren = n.data.children?.length > 0;
          const totalWidth = Math.min(textWidth, 500) + (hasChildren ? 40 : 0) + 24;
          const nodeHeight = textWrapping.height + 10;
          
          return [nodeHeight, totalWidth] as [number, number];
        })
        .spacing(() => nodePadding);
      
      
      return treeLayout(hierarchy) as HierarchyPointNodeWithData;
    }
  })();
  
  
  root.each(d => {
    
    if (overviewMode && d.data.isFiltered) {
      d.nodeWidth = 50;
    } else {
      const textWidth = calculateTextWidth(d.data.token || '');
      const hasChildren = d.data.children?.length > 0;
      d.nodeWidth = textWidth + (hasChildren ? 16 : 0) + 8;
    }
  });
  
  return root;
};


export const calculateEvaluatedPathsProbability = (
  tree: VisualNode | null,
  evaluatedNodeIds: Set<string>
): number => {
  if (!tree || evaluatedNodeIds.size === 0) return 0;
  
  let totalProbability = 0;
  
  
  evaluatedNodeIds.forEach(nodeId => {
    const node = findNodeById(tree, nodeId);
    if (!node) return;
    
    
    let pathProbability = 1.0;
    let currentNode: VisualNode | null = node;
    
    while (currentNode) {
      
      if (currentNode.id !== tree.id) {
        pathProbability *= currentNode.prob;
      }
      
      currentNode = findParentNode(tree, currentNode.id);
    }
    
    totalProbability += pathProbability;
  });
  
  return totalProbability;
};


const findParentNode = (tree: VisualNode, targetId: string): VisualNode | null => {
  
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

