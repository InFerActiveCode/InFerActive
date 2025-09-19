import React, { useState, useMemo } from 'react';
import styled from 'styled-components';
import { VisualNode, EvaluatedNodes} from '../types/types';
import { calculateEvaluatedPathsProbability } from './tokenTreeUtils';
import { findNodeById } from '../utils/treeTransform';

const PanelContainer = styled.div`
  height: 100%;
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const PanelHeader = styled.div`
  padding: 10px 15px;
  background: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  user-select: none;
`;

const PanelTitle = styled.div`
  font-size: 14px;
  font-weight: 600;
  color: #333;
`;

const PanelContent = styled.div`
  padding: 15px;
  flex: 1;
  overflow-y: auto;
`;


const PathList = styled.div`
  margin-top: 0;
`;

const PathItem = styled.div<{ $clickable?: boolean }>`
  padding: 8px;
  margin-bottom: 5px;
  background: #f8f9fa;
  border-radius: 4px;
  font-size: 12px;
  word-break: break-all;
  cursor: ${props => props.$clickable ? 'pointer' : 'default'};
  transition: background-color 0.2s ease;
  
  &:hover {
    background: ${props => props.$clickable ? '#e9ecef' : '#f8f9fa'};
  }
`;

const PathText = styled.div`
  color: #666;
  margin-bottom: 2px;
`;

const PathProb = styled.div`
  color: #2196f3;
  font-weight: 500;
`;


const CategorySection = styled.div`
  margin-bottom: 15px;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const CategoryHeader = styled.div<{ $category: 'good' | 'bad' | 'total' }>`
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
  color: ${props => 
    props.$category === 'good' ? '#2e7d32' : 
    props.$category === 'bad' ? '#c62828' : 
    '#333'};
  display: flex;
  align-items: center;
  gap: 6px;
`;

const CategoryIcon = styled.span<{ $category: 'good' | 'bad' | 'total' }>`
  color: ${props => 
    props.$category === 'good' ? '#4caf50' : 
    props.$category === 'bad' ? '#f44336' : 
    '#2196f3'};
`;

const FilterButton = styled.button<{ $active: boolean }>`
  padding: 4px 8px;
  font-size: 11px;
  background: ${props => props.$active ? '#2196f3' : '#e0e0e0'};
  color: ${props => props.$active ? 'white' : '#666'};
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 4px;
  transition: all 0.2s ease;
  
  &:hover {
    opacity: 0.8;
  }
`;

const RecentSection = styled.div`
  margin-top: 20px;
  padding-top: 15px;
  border-top: 1px solid #e0e0e0;
`;

const RecentHeader = styled.div`
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 10px;
  color: #666;
`;

const RecentItem = styled.div<{ $category: 'good' | 'bad' }>`
  padding: 6px 8px;
  margin-bottom: 4px;
  background: #f8f9fa;
  border-left: 3px solid ${props => 
    props.$category === 'good' ? '#4caf50' : '#f44336'};
  border-radius: 0 4px 4px 0;
  font-size: 11px;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background: #e9ecef;
  }
`;

const NodeToken = styled.div`
  color: #333;
  font-weight: 500;
`;

const NodeId = styled.div`
  color: #999;
  font-size: 10px;
  margin-top: 2px;
`;

interface EvaluatedPathsPanelProps {
  visualTree: VisualNode | null;
  evaluatedNodeIds: EvaluatedNodes | Set<string>;
  onPathSelect?: (nodeId: string) => void;
  filters?: {
    showGood: boolean;
    showBad: boolean;
    showUnmarked: boolean;
  };
  onToggleFilter?: (filter: 'good' | 'bad' | 'unmarked') => void;
}

export const EvaluatedPathsPanel: React.FC<EvaluatedPathsPanelProps> = ({
  visualTree,
  evaluatedNodeIds,
  onPathSelect,
  filters,
  onToggleFilter
}) => {
  
  // Calculate probabilities by category
  const probabilities = useMemo(() => {
    // Handle legacy Set<string> type
    if (evaluatedNodeIds instanceof Set) {
      return {
        total: calculateEvaluatedPathsProbability(visualTree, evaluatedNodeIds),
        good: 0,
        bad: 0
      };
    }
    
    const allNodeIds = new Set([...Array.from(evaluatedNodeIds.good), ...Array.from(evaluatedNodeIds.bad)]);
    const goodNodeIds = evaluatedNodeIds.good;
    const badNodeIds = evaluatedNodeIds.bad;
    
    return {
      total: calculateEvaluatedPathsProbability(visualTree, allNodeIds),
      good: calculateEvaluatedPathsProbability(visualTree, goodNodeIds),
      bad: calculateEvaluatedPathsProbability(visualTree, badNodeIds)
    };
  }, [visualTree, evaluatedNodeIds]);
  
  // Collect path information by category
  const categorizedPaths = useMemo(() => {
    if (!visualTree) return { good: [], bad: [] };
    
    const collectPaths = (nodeIds: Set<string>) => {
      const paths: Array<{ nodeId: string; path: string; probability: number }> = [];
      
      nodeIds.forEach(nodeId => {
      const node = findNodeById(visualTree, nodeId);
      if (!node) return;
      
      // Build path
      const pathTokens: string[] = [];
      let currentNode: VisualNode | null = node;
      let pathProbability = 1.0;
      
      // Traverse from node to root, collecting tokens
      while (currentNode) {
        // Skip root node
        if (currentNode.id !== visualTree.id) {
          pathTokens.unshift(currentNode.token);
        }
        pathProbability *= currentNode.prob;
        
        // Find parent using DFS
        let parent: VisualNode | null = null;
        const findParent = (tree: VisualNode, targetId: string): VisualNode | null => {
          if (tree.children) {
            for (const child of tree.children) {
              if (child.id === targetId) return tree;
              const found = findParent(child, targetId);
              if (found) return found;
            }
          }
          return null;
        };
        
        parent = currentNode.id === visualTree.id ? null : findParent(visualTree, currentNode.id);
        currentNode = parent;
      }
      
        paths.push({
          nodeId,
          path: pathTokens.join(''),
          probability: pathProbability
        });
      });
      
      return paths.sort((a, b) => b.probability - a.probability);
    };
    
    // Handle legacy Set<string> type
    if (evaluatedNodeIds instanceof Set) {
      return { good: collectPaths(evaluatedNodeIds), bad: [] };
    }
    
    return {
      good: collectPaths(evaluatedNodeIds.good),
      bad: collectPaths(evaluatedNodeIds.bad)
    };
  }, [visualTree, evaluatedNodeIds]);
  
  
  return (
    <PanelContainer>
      <PanelHeader>
        <PanelTitle>
          Evaluated Paths - Total: {(probabilities.total * 100).toFixed(1)}%
        </PanelTitle>
        <div style={{ display: 'flex', gap: '4px' }}>
          <FilterButton 
            $active={filters?.showGood ?? true}
            onClick={() => onToggleFilter?.('good')}
          >
            Good
          </FilterButton>
          <FilterButton 
            $active={filters?.showBad ?? true}
            onClick={() => onToggleFilter?.('bad')}
          >
            Bad
          </FilterButton>
        </div>
      </PanelHeader>
      
      <PanelContent>
        {categorizedPaths.good.length === 0 && categorizedPaths.bad.length === 0 && (
            <div style={{ 
              textAlign: 'center', 
              color: '#6c757d', 
              padding: '40px',
              fontSize: '14px'
            }}>
              No evaluated paths yet. Evaluate nodes to see paths here.
            </div>
          )}
          
          {categorizedPaths.good.length > 0 && (
            <CategorySection>
              <CategoryHeader $category="good">
                <CategoryIcon $category="good">✓</CategoryIcon>
                Good: {(probabilities.good * 100).toFixed(1)}%
              </CategoryHeader>
              <PathList>
                {categorizedPaths.good.map(({ nodeId, path, probability }) => (
                  <PathItem 
                    key={nodeId}
                    $clickable={!!onPathSelect}
                    onClick={() => onPathSelect?.(nodeId)}
                  >
                    <PathText>{path}</PathText>
                    <PathProb>{(probability * 100).toFixed(2)}%</PathProb>
                  </PathItem>
                ))}
              </PathList>
            </CategorySection>
          )}
          
          {categorizedPaths.bad.length > 0 && (
            <CategorySection>
              <CategoryHeader $category="bad">
                <CategoryIcon $category="bad">✗</CategoryIcon>
                Bad: {(probabilities.bad * 100).toFixed(1)}%
              </CategoryHeader>
              <PathList>
                {categorizedPaths.bad.map(({ nodeId, path, probability }) => (
                  <PathItem 
                    key={nodeId}
                    $clickable={!!onPathSelect}
                    onClick={() => onPathSelect?.(nodeId)}
                  >
                    <PathText>{path}</PathText>
                    <PathProb>{(probability * 100).toFixed(2)}%</PathProb>
                  </PathItem>
                ))}
              </PathList>
            </CategorySection>
          )}
          
        </PanelContent>
    </PanelContainer>
  );
};